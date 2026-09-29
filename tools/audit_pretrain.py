"""Independently verify materialized corpus hashes, split isolation and token counts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
from prepare_pretrain import digest
from prepare_pretrain_v2 import split_for_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text())
    spec = manifest["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(spec["repo_id"], revision=spec["revision"])
    seen = set()
    report = {}
    for split in ("train", "validation"):
        counts = Counter()
        docs = Counter()
        files = {}
        checksum = hashlib.sha256()
        pending = []
        def check_batch():
            ids = tokenizer([r["text"] for r in pending], add_special_tokens=False, return_attention_mask=False)["input_ids"]
            for record, tokens in zip(pending, ids):
                assert len(tokens) + 1 == record["tokens"], "token count mismatch"
            pending.clear()
        with (args.directory / (split + ".jsonl")).open("rb") as stream:
            for line in stream:
                checksum.update(line)
                record = json.loads(line)
                key = digest(record["text"])
                assert key == record["document_hash"]
                assert key not in seen, "duplicate or split leakage"
                assert split_for_hash(key) == split
                seen.add(key)
                source = record["source"]
                counts[source] += record["tokens"]
                docs[source] += 1
                files.setdefault(source, set()).add(record["origin"]["file"])
                pending.append(record)
                if len(pending) == 64:
                    check_batch()
        if pending:
            check_batch()
        assert checksum.hexdigest() == manifest["outputs"][split]["sha256"]
        assert sum(counts.values()) == manifest["outputs"][split]["tokens"]
        for source in manifest["sources"]:
            assert counts[source] == manifest["sources"][source]["actual"][split]["tokens"]
            assert docs[source] == manifest["sources"][source]["actual"][split]["documents"]
        report[split] = {"tokens": sum(counts.values()), "documents": sum(docs.values()),
                         "tokens_by_source": dict(counts), "shards_by_source": {k: len(v) for k,v in files.items()}}
    report["status"] = "passed"
    report["checks"] = ["full token recount", "file SHA256", "normalized text hashes", "global unique documents", "stable split isolation", "manifest counts"]
    (args.directory / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
