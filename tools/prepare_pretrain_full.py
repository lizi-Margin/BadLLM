#!/usr/bin/env python3
"""Stream the downloaded Parquet corpus into a large, trainable JSONL corpus."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import time
import unicodedata

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from prepare_pretrain import clean_text, digest

REPO = Path(__file__).resolve().parents[1]


def split_for_hash(key: str) -> str:
    return "validation" if int(key[:8], 16) % 100 == 0 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, default=REPO / "data/pretrain/raw")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--train-bytes", type=int, default=50_000_000_000)
    ap.add_argument("--validation-bytes", type=int, default=500_000_000)
    ap.add_argument("--rows-per-batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-doc-chars", type=int, default=200_000)
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output directory is non-empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    mix = json.loads((REPO / "experiments/qwen3-tiny/data_mix.json").read_text())
    tok_spec = json.loads((REPO / "experiments/qwen3-tiny/model_spec.json").read_text())["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tok_spec["repo_id"], revision=tok_spec["revision"])
    plan = json.loads((REPO / "data/pretrain/download_plan.json").read_text())
    provenance = {Path(x["local_path"]).name: x for x in plan["files"]}
    rng = random.Random(args.seed)
    db = sqlite3.connect(args.output / "seen_hashes.sqlite3")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE seen (hash TEXT PRIMARY KEY, split TEXT NOT NULL, source TEXT NOT NULL)")
    outputs = {"train": args.output / "train.jsonl", "validation": args.output / "validation.jsonl"}
    streams = {split: outputs[split].open("w", encoding="utf-8") for split in outputs}
    target_bytes = {"train": args.train_bytes, "validation": args.validation_bytes}
    used_bytes = {split: 0 for split in outputs}
    used_tokens = {split: 0 for split in outputs}
    counts = {s["name"]: {"train": Counter(), "validation": Counter()} for s in mix["sources"]}
    complete = True
    started = time.time()
    try:
        for source in mix["sources"]:
            name, column = source["name"], source["text_column"]
            source_targets = {"train": int(args.train_bytes * source["token_weight"]),
                              "validation": int(args.validation_bytes * source["token_weight"])}
            source_tokens = {"train": 0, "validation": 0}
            files = sorted((args.raw_root / name).glob("*.parquet"))
            rng.shuffle(files)
            source_complete = False
            for path in files:
                if all(source_tokens[s] >= source_targets[s] for s in source_targets):
                    source_complete = True
                    break
                parquet = pq.ParquetFile(path)
                groups = list(range(parquet.num_row_groups))
                rng.shuffle(groups)
                for group in groups:
                    if all(source_tokens[s] >= source_targets[s] for s in source_targets):
                        source_complete = True
                        break
                    names = parquet.schema_arrow.names
                    columns = [column] + [x for x in ("url", "id", "title") if x in names and x != column]
                    batch = parquet.read_row_group(group, columns=columns)
                    values = batch.column(column).to_pylist()
                    candidates, origins = [], []
                    for row_index, value in enumerate(values):
                        text = clean_text(value)
                        if not text or len(text) > args.max_doc_chars:
                            continue
                        key = digest(text)
                        split = split_for_hash(key)
                        if source_tokens[split] >= source_targets[split] or used_bytes[split] >= target_bytes[split]:
                            continue
                        candidates.append(text)
                        origins.append({"file": path.name, "row_group": group, "row_in_group": row_index,
                            "repo_id": source["repo_id"], "revision": source["revision"],
                            "remote_path": provenance.get(path.name, {}).get("remote_path", path.name),
                            "raw_sha256": provenance.get(path.name, {}).get("sha256", "unknown")})
                    if candidates:
                        encoded = tokenizer(candidates, add_special_tokens=False, return_attention_mask=False)["input_ids"]
                        for text, origin, ids in zip(candidates, origins, encoded):
                            key = digest(text)
                            split = split_for_hash(key)
                            if source_tokens[split] >= source_targets[split] or used_bytes[split] >= target_bytes[split] or len(ids) < 32:
                                continue
                            record = {"text": text, "source": name, "document_hash": key,
                                      "tokens": len(ids) + 1, "origin": origin}
                            raw = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                            raw_bytes = len(raw.encode("utf-8"))
                            inserted = db.execute("INSERT OR IGNORE INTO seen VALUES (?, ?, ?)", (key, split, name)).rowcount
                            if not inserted:
                                continue
                            streams[split].write(raw)
                            source_tokens[split] += len(ids) + 1
                            used_tokens[split] += len(ids) + 1
                            used_bytes[split] += raw_bytes
                            counts[name][split]["documents"] += 1
                            counts[name][split]["tokens"] += len(ids) + 1
                            counts[name][split]["bytes"] += raw_bytes
                    db.commit()
                    if group % 10 == 0:
                        elapsed = max(1, time.time() - started)
                        print(json.dumps({"source": name, "train_gb": used_bytes["train"] / 1e9,
                            "validation_mb": used_bytes["validation"] / 1e6, "tokens": used_tokens,
                            "rate_mb_s": used_bytes["train"] / 1e6 / elapsed}, ensure_ascii=False), flush=True)
                if source_complete:
                    break
            complete &= source_complete
            print(json.dumps({"source": name, "quota_reached": source_complete,
                "source_tokens": source_tokens}, ensure_ascii=False), flush=True)
    finally:
        for stream in streams.values():
            stream.close()
        db.close()
    output_meta = {}
    for split, path in outputs.items():
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        output_meta[split] = {"bytes": path.stat().st_size, "tokens": used_tokens[split], "sha256": h.hexdigest()}
    manifest = {"schema_version": 1, "status": "complete" if complete else "source_quota_unreached",
        "filter_version": "full-stream-v1", "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "tokenizer": tok_spec,
        "outputs": output_meta, "sources": {k: {s: dict(v) for s, v in d.items()} for k, d in counts.items()},
        "dedup": "global normalized-text SHA256 in SQLite", "split": "stable hash modulo 100; validation is disjoint",
        "limitations": ["Near-duplicate clustering and benchmark contamination screening are not performed.",
                        "raw_sha256 is populated from the download plan in the next provenance audit."]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "dataset_info.json").write_text(json.dumps({"full_train": {"file_name": "train.jsonl", "columns": {"prompt": "text"}}}, indent=2))
    print(json.dumps({"status": manifest["status"], "outputs": output_meta}, ensure_ascii=False, indent=2), flush=True)
    if not complete:
        raise SystemExit("one or more source quotas were not reached")


if __name__ == "__main__":
    main()
