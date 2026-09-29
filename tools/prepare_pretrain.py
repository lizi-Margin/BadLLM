#!/usr/bin/env python3
"""Materialize a small, reproducible pretraining corpus from local Parquet shards.

The script deliberately writes document JSONL (rather than token IDs) so that
LLaMA-Factory remains responsible for packing and label construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer


SOURCES = {
    "ultrafineweb_zh": {"column": "content", "weight": 0.40},
    "fineweb_edu_dedup": {"column": "text", "weight": 0.35},
    "cosmopedia_v2": {"column": "text", "weight": 0.15},
    "finewiki_zh": {"column": "text", "weight": 0.05},
    "finemath_4plus": {"column": "text", "weight": 0.05},
}


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    if len(text) < 80 or len(text) > 200_000:
        return ""
    # Reject replacement-heavy or binary-looking rows.
    if text.count("\ufffd") > 2 or (text.count("\x00") > 0):
        return ""
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in text)
    if printable / max(1, len(text)) < 0.97:
        return ""
    return text


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, default=Path("data/pretrain/raw"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    ap.add_argument("--tokens", type=int, default=10_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-batches-per-source", type=int, default=100000)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision, use_fast=True)
    args.output.mkdir(parents=True, exist_ok=True)
    train_path = args.output / "train.jsonl"
    valid_path = args.output / "validation.jsonl"
    manifest_path = args.output / "manifest.json"
    # Allocate an exact target per source; rounding remainder goes to the largest source.
    targets = {name: int(args.tokens * spec["weight"]) for name, spec in SOURCES.items()}
    targets["ultrafineweb_zh"] += args.tokens - sum(targets.values())
    seen: set[str] = set()
    validation_seen: set[str] = set()
    stats = {name: {"documents": 0, "tokens": 0, "filtered": 0, "duplicates": 0} for name in SOURCES}
    all_files: dict[str, list[Path]] = {}
    for name in SOURCES:
        files = sorted((args.raw_root / name).glob("*.parquet"))
        rng.shuffle(files)
        if not files:
            raise FileNotFoundError(f"no parquet files for {name}")
        all_files[name] = files

    with train_path.open("w", encoding="utf-8") as train, valid_path.open("w", encoding="utf-8") as valid:
        # Reserve one deterministic validation document out of every 100 accepted documents.
        for source, spec in SOURCES.items():
            batches_seen = 0
            for parquet_path in all_files[source]:
                if stats[source]["tokens"] >= targets[source]:
                    break
                parquet = pq.ParquetFile(parquet_path)
                for batch in parquet.iter_batches(batch_size=args.batch_size, columns=[spec["column"]]):
                    batches_seen += 1
                    candidates = []
                    for value in batch.column(0).to_pylist():
                        text = clean_text(value)
                        if not text:
                            stats[source]["filtered"] += 1
                            continue
                        key = digest(text)
                        if key in seen:
                            stats[source]["duplicates"] += 1
                            continue
                        candidates.append((text, key))
                    if not candidates:
                        continue
                    encoded = tokenizer([x[0] for x in candidates], add_special_tokens=False, return_attention_mask=False)
                    for (text, key), ids in zip(candidates, encoded["input_ids"]):
                        if key in seen:
                            stats[source]["duplicates"] += 1
                            continue
                        token_count = len(ids) + 1
                        if token_count < 32:
                            stats[source]["filtered"] += 1
                            continue
                        # Keep validation separate from the training budget.
                        record = {"text": text, "source": source, "document_hash": key, "tokens": token_count}
                        if (int(key[:8], 16) % 100) == 0:
                            if key not in validation_seen:
                                valid.write(json.dumps(record, ensure_ascii=False) + "\n")
                                validation_seen.add(key)
                            seen.add(key)
                            continue
                        if stats[source]["tokens"] >= targets[source]:
                            continue
                        train.write(json.dumps(record, ensure_ascii=False) + "\n")
                        train.flush()
                        seen.add(key)
                        stats[source]["documents"] += 1
                        stats[source]["tokens"] += token_count
                    if stats[source]["tokens"] >= targets[source]:
                        break
                    if batches_seen >= args.max_batches_per_source:
                        break
                if stats[source]["tokens"] >= targets[source]:
                    break
                if batches_seen >= args.max_batches_per_source:
                    break

    manifest = {
        "schema_version": 1,
        "status": "materialized",
        "tokenizer": {"repo_id": args.tokenizer, "revision": args.revision, "vocab_size": tokenizer.vocab_size},
        "seed": args.seed,
        "requested_tokens": args.tokens,
        "targets_by_source": targets,
        "stats_by_source": stats,
        "train_documents": sum(x["documents"] for x in stats.values()),
        "train_tokens": sum(x["tokens"] for x in stats.values()),
        "validation_documents": len(validation_seen),
        "source_files": {k: [str(p) for p in v] for k, v in all_files.items()},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
