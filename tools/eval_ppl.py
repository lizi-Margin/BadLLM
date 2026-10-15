"""Per-domain perplexity for any checkpoint (works on CUDA and MUSA boxes).

Eval files are jsonl with a `text` field, one file per domain (as produced by
build_hq_dataset.py eval_<source>.jsonl). Each domain is tokenized, chunked into
1024-token blocks and scored with teacher forcing; PPL = exp(NLL/token).

Usage:
  python tools/eval_ppl.py --model wsl/runs/qwen3-tiny/pt-3b-v5-after-v4 \
      --data data/pretrain/prepared/pt-hq-2b --max-tokens 200000 --out eval.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 1024


def resolve_device(name: str):
    import sys
    import torch
    if name != 'auto':
        return torch.device(name)
    try:
        import torch_musa  # noqa: F401
        try:
            import musa_compat  # noqa: F401  (site-packages shims, required on MUSA)
        except ImportError:
            # Fall back to the vendored copy so a fresh clone works without
            # installing the shim into the environment.
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import musa_compat  # noqa: F401
        if torch.musa.is_available():
            return torch.device('musa')
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def domain_batches(path: Path, tokenizer, max_tokens: int):
    ids: list[int] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            ids.extend(tokenizer(rec['text'], add_special_tokens=False)['input_ids'])
            ids.append(tokenizer.eos_token_id)
            while len(ids) >= BLOCK:
                yield ids[:BLOCK]
                ids = ids[BLOCK:]
                max_tokens -= BLOCK
                if max_tokens <= 0:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True,
                    help='dir with eval_*.jsonl or a single jsonl file')
    ap.add_argument('--max-tokens', type=int, default=200000, help='per domain')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float32'])
    ap.add_argument('--out', type=Path, help='write json results here')
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    files = sorted(a.data.glob('eval_*.jsonl')) if a.data.is_dir() else [a.data]
    files = [f for f in files if f.name != 'eval_mixed.jsonl'] or files
    if not files:
        raise FileNotFoundError(f'no eval jsonl under {a.data}')
    device = resolve_device(a.device)
    dtype = torch.bfloat16 if a.dtype == 'bfloat16' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, trust_remote_code=True)
    model.to(device).eval()
    print(f'model={a.model} device={device}', flush=True)

    results = {}
    for path in files:
        nll_sum, n_tokens, n_blocks = 0.0, 0, 0
        batch: list[list[int]] = []

        def consume(batch: list[list[int]]) -> None:
            nonlocal nll_sum, n_tokens, n_blocks
            if not batch:
                return
            input_ids = torch.tensor(batch, dtype=torch.long, device=device)
            labels = input_ids.clone()
            with torch.inference_mode():
                logits = model(input_ids=input_ids).logits
            nll = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1), reduction='sum')
            nll_sum += float(nll)
            n_tokens += labels.numel() - len(batch)
            n_blocks += len(batch)
            batch.clear()

        for chunk in domain_batches(path, tokenizer, a.max_tokens):
            batch.append(chunk)
            if len(batch) >= a.batch_size:
                consume(batch)
                if n_blocks % 20 == 0:
                    print(f'  {path.name}: {n_tokens} tokens, running ppl '
                          f'{np.exp(nll_sum / max(1, n_tokens)):.3f}', flush=True)
        consume(batch)
        ppl = float(np.exp(nll_sum / max(1, n_tokens)))
        domain = path.stem.replace('eval_', '') or path.stem
        results[domain] = dict(file=str(path), tokens=n_tokens, blocks=n_blocks, ppl=round(ppl, 4))
        print(json.dumps({domain: results[domain]}), flush=True)

    weights = [r['tokens'] for r in results.values()]
    macro = float(np.mean([r['ppl'] for r in results.values()]))
    micro = float(np.exp(sum(np.log(r['ppl']) * w for r, w in zip(results.values(), weights)) / max(1, sum(weights))))
    summary = dict(model=str(a.model), device=str(device), dtype=a.dtype,
                   per_domain=results, macro_ppl=round(macro, 4), token_weighted_ppl=round(micro, 4))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if a.out:
        a.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
