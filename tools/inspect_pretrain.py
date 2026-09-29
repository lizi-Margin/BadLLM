"""Read real, verified Parquet samples without loading full shards into memory.

Run with .venv-data/Scripts/python.exe on Windows (requires pyarrow).
"""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from download_pretrain import DEFAULT_ROOT, is_verified, local_path, now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "download_plan.json").read_text(encoding="utf-8"))
    sources = {s["name"]: s for s in plan["sources"]}
    result = {"checked_at": now(), "scope": "One verified shard per source; up to 32 real rows each.", "sources": {}}
    for entry in plan["files"]:
        name = entry["source"]
        if name in result["sources"]:
            continue
        path = local_path(root, entry)
        if not is_verified(path, entry):
            continue
        parquet = pq.ParquetFile(path)
        column = sources[name]["text_column"]
        if column not in parquet.schema_arrow.names:
            raise ValueError(f"{path}: missing text column {column}")
        batch = next(parquet.iter_batches(batch_size=32, columns=[column]))
        texts = batch.column(0).to_pylist()
        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError(f"{path}: invalid text rows")
        info = {
            "file": entry["local_path"], "revision": entry["revision"],
            "sha256": entry["sha256"], "parquet_rows": parquet.metadata.num_rows,
            "row_groups": parquet.metadata.num_row_groups, "columns": parquet.schema_arrow.names,
            "text_column": column, "sample_rows": len(texts),
            "sample_empty_rows": sum(not t.strip() for t in texts),
            "sample_min_characters": min(map(len, texts)),
            "sample_max_characters": max(map(len, texts)),
            "sample_first_160_characters": texts[0][:160],
        }
        result["sources"][name] = info
        print(f"{name}: {info['parquet_rows']:,} rows, text={column}, "
              f"sample={len(texts)}, empty={info['sample_empty_rows']}", flush=True)
    result["pending_sources"] = [s for s in sources if s not in result["sources"]]
    write_json(root / "sample_inspection.json", result)
    if result["pending_sources"]:
        print("Pending verified shards: " + ", ".join(result["pending_sources"]))


if __name__ == "__main__":
    main()
