#!/usr/bin/env python3
"""Evaluate causal-LM loss on the materialized pretraining validation JSONL."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate(model_path: str, data_path: Path, tokenizer_path: str, revision: str, batch_size: int):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    by_source = defaultdict(lambda: {"nll": 0.0, "tokens": 0, "documents": 0})
    batch_ids, batch_sources = [], []
    def flush():
        if not batch_ids:
            return
        width = max(len(x) for x in batch_ids)
        pad = tokenizer.pad_token_id
        ids = torch.full((len(batch_ids), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(batch_ids):
            ids[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
        ids, mask = ids.cuda(), mask.cuda()
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        labels = ids[:, 1:]
        valid = mask[:, 1:].bool()
        losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        for i, source in enumerate(batch_sources):
            n = int(valid[i].sum())
            value = float(losses[i][valid[i]].sum())
            by_source[source]["nll"] += value
            by_source[source]["tokens"] += n
            by_source[source]["documents"] += 1
        batch_ids.clear(); batch_sources.clear()
    with data_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            ids = tokenizer.encode(record["text"], add_special_tokens=False) + [tokenizer.eos_token_id]
            # Evaluate every document as contiguous 1024-token windows.
            for start in range(0, len(ids) - 1, 1024):
                row = ids[start:start + 1024]
                if len(row) < 2:
                    continue
                batch_ids.append(row); batch_sources.append(record["source"])
                if len(batch_ids) >= batch_size:
                    flush()
            # Document count is tracked per window in this simple evaluator.
    flush()
    result = {}
    total_nll = total_tokens = 0
    for source, values in by_source.items():
        result[source] = {**values, "loss": values["nll"] / values["tokens"],
                          "perplexity": math.exp(min(20, values["nll"] / values["tokens"]))}
        total_nll += values["nll"]; total_tokens += values["tokens"]
    result["overall"] = {"nll": total_nll, "tokens": total_tokens,
                          "loss": total_nll / total_tokens,
                          "perplexity": math.exp(min(20, total_nll / total_tokens))}
    del model
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = evaluate(args.model, args.data, args.tokenizer, args.revision, args.batch_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
