"""Build a small high-quality pretraining dataset by scoring and selecting from
existing prepared exports (build_pretrain_v3/v5 train/validation jsonl).

Scaling-law rationale: the 114M model saw 17B tokens (bulk web heavy) and the
final loss plateaued near 2.92 with no coherence gain. This builder inverts the
strategy: pick ~2B tokens of the highest-quality documents from the ~6B tokens
already prepared, with per-source quotas favoring encyclopedic, educational and
synthetic-textbook text over bulk web.

Quality score (cheap heuristics only, rank WITHIN each source):
  - duplicated-line ratio (boilerplate / copy-paste junk)
  - repeated n-gram density (list spam, lorem, templated repetition)
  - unique-character ratio (garbled or degenerate text)
  - line-length spread (structured prose > walls of text)

Outputs a LLaMA-Factory-ready directory:
  train.jsonl               slim records {text, source, document_hash, tokens}
  eval_<source>.jsonl       held-out per-domain eval docs (from pool validation.jsonl)
  eval_mixed.jsonl          union of per-domain eval docs
  dataset_info.json         LLaMA-Factory dataset registration
  manifest.json             per-source available/selected stats and score summary

Windows/WSL (training box):
  python tools/build_hq_dataset.py ^
    --mix experiments/qwen3-tiny/data_mix_hq.json ^
    --output data/pretrain/prepared/pt-hq-2b ^
    --pool data/pretrain/prepared/pt-3b-v4 ^
    --pool data/pretrain/prepared/pt-3b-v5-reasoning
"""
from __future__ import annotations
import argparse
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'hq-select-v1'


def quality_score(text: str) -> tuple[float, dict]:
    """Higher is better. Returns (score, features) for manifest aggregation."""
    lines = [ln for ln in text.split('\n') if len(ln) >= 10]
    dup_line_ratio = 0.0
    if len(lines) >= 4:
        dup_line_ratio = 1.0 - len(set(lines)) / len(lines)
    window = text[:20000]
    n = 8  # character 8-grams work uniformly for zh and en
    grams = Counter(window[i:i + n] for i in range(0, max(1, len(window) - n), 4))
    repeated_gram_ratio = 0.0
    if len(grams) >= 32:
        repeated_gram_ratio = 1.0 - len(set(grams)) / len(grams)
    unique_char_ratio = len(set(text)) / max(1, len(text))
    lengths = [len(ln) for ln in text.split('\n') if ln.strip()]
    spread = 0.0
    if len(lengths) >= 2:
        mean = sum(lengths) / len(lengths)
        spread = min(1.0, (sum((x - mean) ** 2 for x in lengths) / len(lengths)) ** .5 / max(1.0, mean))
    score = 1.0
    score -= 2.5 * dup_line_ratio
    score -= 2.0 * repeated_gram_ratio
    score -= 1.0 * max(0.0, 0.35 - unique_char_ratio) * 4.0
    score += 0.2 * spread
    feats = dict(dup_line=round(dup_line_ratio, 4), rep_gram=round(repeated_gram_ratio, 4),
                 uniq_char=round(unique_char_ratio, 4), spread=round(spread, 4))
    return score, feats


def stream_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mix', type=Path, default=ROOT / 'experiments/qwen3-tiny/data_mix_hq.json')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--pool', type=Path, action='append', required=True,
                    help='prepared dataset dir containing train.jsonl / validation.jsonl (repeatable)')
    ap.add_argument('--eval-docs-per-source', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    mix = json.loads(args.mix.read_text(encoding='utf-8'))
    targets = {s['name']: int(s['tokens']) for s in mix['sources']}
    eval_cap = int(mix.get('eval_max_tokens_per_source', 500000))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pools = [p.resolve() for p in args.pool]
    for p in pools:
        for f in ('train.jsonl', 'validation.jsonl'):
            if not (p / f).exists():
                raise FileNotFoundError(f'{p / f} missing; pass prepared dataset dirs')

    signature = hashlib.sha256(json.dumps(
        dict(version=VERSION, mix=mix, pools=[str(p) for p in pools], seed=args.seed),
        sort_keys=True).encode()).hexdigest()

    # ---- pass 1: score every train doc, reservoir-sample eval docs -------------
    index = defaultdict(list)          # source -> [(score, tokens, hash), ...]
    eval_docs = defaultdict(list)      # source -> [(-priority, tokens, hash, text)]
    stats = defaultdict(Counter)
    import random
    rng = random.Random(args.seed)
    for pool in pools:
        vpath = pool / 'validation.jsonl'
        eval_priority = {}
        for rec in stream_jsonl(vpath):
            src = rec.get('source')
            if src not in targets:
                continue
            tokens = int(rec.get('tokens', 0)) or None
            if tokens is None:
                continue
            pri = rng.random()
            bucket = (-pri, tokens, rec['document_hash'], rec['text'])
            if len(eval_docs[src]) < args.eval_docs_per_source:
                heapq.heappush(eval_docs[src], bucket)
            elif pri > -eval_docs[src][0][0]:
                heapq.heapreplace(eval_docs[src], bucket)
            eval_priority[rec['document_hash']] = None  # reserved, never trained on
        for rec in stream_jsonl(pool / 'train.jsonl'):
            src = rec.get('source')
            if src not in targets or rec['document_hash'] in eval_priority:
                stats[src]['in_validation_pool'] += 1
                continue
            text = rec.get('text')
            if not isinstance(text, str) or not text:
                stats[src]['empty_text'] += 1
                continue
            score, _ = quality_score(text)
            index[src].append((score, int(rec.get('tokens', 0)), rec['document_hash']))
            stats[src]['scored'] += 1
        print(f'scored pool {pool.name}: ' + json.dumps(
            {s: dict(stats[s]) for s in sorted(index)}, ensure_ascii=False), flush=True)

    # ---- selection: top-scored docs per source until token quota ---------------
    selected = {}                                # hash -> None
    summary = {}
    for src, target in targets.items():
        docs = index.get(src, [])
        docs.sort(key=lambda x: -x[0])
        picked, used, scores = [], 0, []
        for score, tokens, h in docs:
            if used >= target or h in selected:
                continue
            picked.append(h)
            selected[h] = None
            used += tokens
            scores.append(score)
        scores.sort()
        summary[src] = dict(
            available_tokens=sum(t for _, t, _ in docs), available_docs=len(docs),
            selected_tokens=used, selected_docs=len(picked),
            quota=target, capped=used < target,
            score_min=round(scores[0], 4) if scores else None,
            score_median=round(scores[len(scores) // 2], 4) if scores else None,
            score_max=round(scores[-1], 4) if scores else None)
        print(json.dumps({src: summary[src]}, ensure_ascii=False), flush=True)
        if used < target:
            print(f'WARNING: source {src} exhausted at {used}/{target} tokens', flush=True)

    # ---- pass 2: emit slim train.jsonl -----------------------------------------
    train_path = out / 'train.jsonl'
    total_tokens = 0
    with train_path.open('w', encoding='utf-8') as f:
        for pool in pools:
            for rec in stream_jsonl(pool / 'train.jsonl'):
                h = rec.get('document_hash')
                if h in selected:
                    slim = dict(text=rec['text'], source=rec['source'], document_hash=h,
                                tokens=int(rec.get('tokens', 0)))
                    f.write(json.dumps(slim, ensure_ascii=False, separators=(',', ':')) + '\n')
                    total_tokens += slim['tokens']
                    del selected[h]
    leftover = len(selected)
    if leftover:
        raise RuntimeError(f'{leftover} selected hashes never emitted from pools')

    # ---- per-domain eval files from validation pools ---------------------------
    eval_files, eval_tokens_total = [], 0
    for src in sorted(eval_docs):
        docs = sorted(eval_docs[src])            # deterministic order
        path = out / f'eval_{src}.jsonl'
        used = 0
        with path.open('w', encoding='utf-8') as f:
            for _, tokens, h, text in docs:
                if used >= eval_cap:
                    break
                f.write(json.dumps(dict(text=text, source=src, document_hash=h), ensure_ascii=False,
                                   separators=(',', ':')) + '\n')
                used += tokens
        eval_tokens_total += used
        eval_files.append(dict(source=src, file=path.name, tokens=used, docs=min(len(docs), args.eval_docs_per_source)))
    mixed = out / 'eval_mixed.jsonl'
    with mixed.open('w', encoding='utf-8') as f:
        for src in sorted(eval_docs):
            docs = sorted(eval_docs[src])
            used = 0
            for _, tokens, h, text in docs:
                if used >= eval_cap:
                    break
                f.write(json.dumps(dict(text=text, source=src, document_hash=h), ensure_ascii=False,
                                   separators=(',', ':')) + '\n')
                used += tokens

    dataset_info = {'hq_train': {'file_name': 'train.jsonl', 'columns': {'prompt': 'text'}}}
    for entry in eval_files:
        dataset_info[f"eval_{entry['source']}"] = {
            'file_name': entry['file'], 'columns': {'prompt': 'text'}}
    (out / 'dataset_info.json').write_text(json.dumps(dataset_info, indent=2), encoding='utf-8')

    manifest = dict(schema_version=1, status='complete', builder_version=VERSION, signature=signature,
                    mix=mix, pools=[str(p) for p in pools], seed=args.seed,
                    train_tokens=total_tokens, train_file='train.jsonl',
                    eval=eval_files, eval_tokens=eval_tokens_total,
                    eval_note='per-domain held-out docs sampled from pool validation.jsonl (hash-disjoint from train selection)',
                    scoring='quality_score: duplicated-line ratio, char-8gram repetition, unique-char ratio, line spread; ranked within source',
                    per_source=summary, created_at=__import__('time').time())
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(status='complete', output=str(out), train_tokens=total_tokens,
                          eval_files=len(eval_files), manifest=str(out / 'manifest.json')), indent=2), flush=True)


if __name__ == '__main__':
    main()
