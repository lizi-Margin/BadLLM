"""Reproducible, bounded-memory pilot sampling with provenance and exact dedup."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sqlite3

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from prepare_pretrain import clean_text, digest

REPO = Path(__file__).resolve().parents[1]


def split_for_hash(key):
    # Preserve the v1 holdout assignment across budgets, sources and shuffle seeds.
    return "validation" if int(key[:8], 16) % 100 == 0 else "train"


def batches(files, column, rng, sample_size):
    readers = []
    try:
        for path in files:
            reader = pq.ParquetFile(path)
            groups = list(range(reader.num_row_groups))
            rng.shuffle(groups)
            metadata = json.loads(Path(str(path) + ".verified.json").read_text())
            readers.append((path, reader, groups, metadata))
        while any(groups for _, _, groups, _ in readers):
            for path, reader, groups, metadata in readers:
                if not groups:
                    continue
                group = groups.pop()
                columns = [column] + [x for x in ("url", "id", "title") if x in reader.schema_arrow.names and x != column]
                table = reader.read_row_group(group, columns=columns)
                indices = rng.sample(range(len(table)), min(sample_size, len(table)))
                rows = table.take(indices).to_pylist()
                del table
                yield [(row[column], {"file": path.name, "row_group": group, "row_in_group": index,
                    "repo_id": metadata["repo_id"], "revision": metadata["revision"],
                    "remote_path": metadata["remote_path"], "raw_sha256": metadata["sha256"],
                    "metadata": {k: str(v) for k, v in row.items() if k != column}})
                    for index, row in zip(indices, rows)]
    finally:
        for _, reader, _, _ in readers:
            reader.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, default=REPO / "data/pretrain/raw")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=10_000_000)
    ap.add_argument("--validation-tokens", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rows-per-group", type=int, default=512)
    args = ap.parse_args()
    if min(args.tokens, args.validation_tokens, args.rows_per_group) <= 0:
        ap.error("budgets and rows-per-group must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = json.loads((REPO / "experiments/qwen3-tiny/data_mix.json").read_text())["sources"]
    spec = json.loads((REPO / "experiments/qwen3-tiny/model_spec.json").read_text())["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(spec["repo_id"], revision=spec["revision"])
    special_ids = set(tokenizer.all_special_ids)
    rng = random.Random(args.seed)
    db = sqlite3.connect(args.output / "documents.sqlite3")
    db.execute("CREATE TABLE docs (hash TEXT PRIMARY KEY, split TEXT, shuffle INTEGER, record TEXT)")
    stats = {}
    complete = True
    for source in sources:
        name = source["name"]
        targets = {"train": round(args.tokens * source["token_weight"]),
                   "validation": round(args.validation_tokens * source["token_weight"])}
        counts = {s: {"tokens": 0, "documents": 0} for s in targets}
        filtered = Counter()
        files_used = set()
        files = sorted((args.raw_root / name).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(name)
        rng.shuffle(files)
        groups_read = 0
        stream = batches(files, source["text_column"], rng, args.rows_per_group)
        try:
            for batch in stream:
                groups_read += 1
                candidates = []
                for value, origin in batch:
                    text = clean_text(value)
                    if not text:
                        filtered["basic_quality"] += 1
                        continue
                    lines = [x.strip() for x in text.splitlines() if x.strip()]
                    if sum(map(len, set(lines))) / sum(map(len, lines)) < .5:
                        filtered["repeated_lines"] += 1
                        continue
                    key = digest(text)
                    split = split_for_hash(key)
                    if counts[split]["tokens"] < targets[split]:
                        candidates.append((text, key, split, origin))
                if candidates:
                    encoded = tokenizer([x[0] for x in candidates], add_special_tokens=False,
                                        return_attention_mask=False)["input_ids"]
                    for (text, key, split, origin), ids in zip(candidates, encoded):
                        if counts[split]["tokens"] >= targets[split]:
                            continue
                        if len(ids) < 32 or special_ids.intersection(ids):
                            filtered["short_or_special_tokens"] += 1
                            continue
                        record = {"text": text, "source": name, "document_hash": key,
                                  "tokens": len(ids) + 1, "origin": origin}
                        inserted = db.execute("INSERT OR IGNORE INTO docs VALUES (?, ?, ?, ?)",
                            (key, split, rng.getrandbits(62), json.dumps(record, ensure_ascii=False))).rowcount
                        if not inserted:
                            filtered["duplicate"] += 1
                            continue
                        counts[split]["tokens"] += len(ids) + 1
                        counts[split]["documents"] += 1
                        files_used.add(origin["file"])
                db.commit()
                print(name, groups_read, counts, flush=True)
                if all(counts[s]["tokens"] >= targets[s] for s in targets):
                    break
        finally:
            stream.close()
        reached = all(counts[s]["tokens"] >= targets[s] for s in targets)
        complete &= reached
        stats[name] = {"targets": targets, "actual": counts, "filtered": dict(filtered),
                       "groups_read": groups_read, "files_used": sorted(files_used), "quota_reached": reached,
                       "license_note": source["license_note"]}
    outputs = {}
    for split in ("train", "validation"):
        output = args.output / (split + ".jsonl")
        checksum = hashlib.sha256()
        with output.open("wb") as stream:
            for (record,) in db.execute("SELECT record FROM docs WHERE split=? ORDER BY shuffle,hash", (split,)):
                payload = (record + "\n").encode("utf-8")
                stream.write(payload)
                checksum.update(payload)
        outputs[split] = {"sha256": checksum.hexdigest(), "bytes": output.stat().st_size,
                          "tokens": sum(s["actual"][split]["tokens"] for s in stats.values()),
                          "documents": sum(s["actual"][split]["documents"] for s in stats.values())}
    db.close()
    info = {"pilot_" + split: {"file_name": split + ".jsonl", "columns": {"prompt": "text"}} for split in outputs}
    (args.output / "dataset_info.json").write_text(json.dumps(info, indent=2))
    manifest = {"schema_version": 2, "filter_version": "pilot-v2", "status": "complete" if complete else "insufficient_sample_pool",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "tokenizer": spec, "outputs": outputs, "sources": stats,
        "quota_policy": "Keep whole final document; at most one-document overshoot per source/split.",
        "token_accounting": "Text tokens plus one document EOS, before packing and tail loss.",
        "limitations": ["Exact document dedup only; near-duplicate clustering not performed.",
                         "Benchmark contamination screening not performed.",
                         "Random row-group/row sampling, not a full-corpus quality census."]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(outputs, indent=2), flush=True)
    if not complete:
        raise SystemExit("Sample pool exhausted before quotas; see manifest. No automatic repetition.")


if __name__ == "__main__":
    main()
