"""Append verified, previously unselected HF Parquet shards to the raw plan."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import re
import urllib.request

REPO = Path(__file__).resolve().parents[1]
PREFIXES = {
    "ultrafineweb_zh": "data/ultrafineweb_zh",
    "fineweb_edu_dedup": "fineweb-edu-dedup",
    "cosmopedia_v2": "cosmopedia-v2",
    "finewiki_zh": "data/zhwiki",
    "finemath_4plus": "finemath-4plus",
}


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "BadLLM-corpus-planner/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response), response.headers.get("Link", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "data/pretrain")
    ap.add_argument("--target-gb", type=float, default=80)
    args = ap.parse_args()
    plan_path = args.root / "download_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    existing = {x["remote_path"] for x in plan["files"]}
    current = sum(x["bytes"] for x in plan["files"])
    if current >= args.target_gb * 1e9:
        print(f"already at {current / 1e9:.2f} GB")
        return
    mix = json.loads((REPO / "experiments/qwen3-tiny/data_mix.json").read_text(encoding="utf-8"))
    extra_needed = args.target_gb * 1e9 - current
    # Preserve the selected token-mix proportions approximately.
    selected = []
    for source in mix["sources"]:
        name = source["name"]
        url = f"https://huggingface.co/api/datasets/{source['repo_id']}/tree/{source['revision']}/{PREFIXES[name]}?limit=1000"
        catalog = []
        while url:
            entries, link = get_json(url)
            catalog.extend(e for e in entries if e.get("type") == "file" and e.get("path", "").endswith(".parquet"))
            match = re.search(r'<([^>]+)>;\s*rel="next"', link)
            url = match.group(1) if match else None
        rng = random.Random(f"extension:{plan['seed']}:{name}")
        rng.shuffle(catalog)
        source_target = extra_needed * source["token_weight"]
        source_bytes = 0
        for entry in catalog:
            path = entry["path"]
            if path in existing:
                continue
            sha = entry.get("lfs", {}).get("oid", "")
            if not re.fullmatch(r"[0-9a-f]{64}", sha):
                continue
            row = {"source": name, "repo_id": source["repo_id"], "revision": source["revision"],
                   "remote_path": path, "bytes": entry["size"], "sha256": sha,
                   "local_path": f"raw/{name}/{Path(path).name}",
                   "url": f"https://huggingface.co/datasets/{source['repo_id']}/resolve/{source['revision']}/{path}"}
            selected.append(row); existing.add(path); source_bytes += entry["size"]
            if source_bytes >= source_target:
                break
    plan["files"].extend(selected)
    plan["total_bytes"] = sum(x["bytes"] for x in plan["files"])
    plan["note"] = "Raw acquisition budget, extended to approximately 80GB; not final token proportions or cleaned data."
    plan["extension"] = {"target_gb": args.target_gb, "added_files": len(selected),
                          "added_bytes": sum(x["bytes"] for x in selected)}
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan["extension"], indent=2))


if __name__ == "__main__":
    main()
