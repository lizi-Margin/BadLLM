"""Windows-native, resumable, SHA-256 verified HF dataset shard downloads.

Uses Python's standard library. Run `plan`, then `run`; `status` is read-only.
Download paths and dataset revisions are fixed in download_plan.json.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "data" / "pretrain"
PREFIXES = {
    "ultrafineweb_zh": ("data/ultrafineweb_zh", 16_000_000_000),
    "fineweb_edu_dedup": ("fineweb-edu-dedup", 14_000_000_000),
    "cosmopedia_v2": ("cosmopedia-v2", 6_000_000_000),
    "finewiki_zh": ("data/zhwiki", None),
    "finemath_4plus": ("finemath-4plus", 2_000_000_000),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def open_url(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": "BadLLM-corpus-downloader/1.0", "Accept-Encoding": "identity", **(headers or {})}
    ), timeout=60)


def fetch_json(url):
    for attempt in range(5):
        try:
            with open_url(url) as response:
                return json.load(response), response.headers.get("Link", "")
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def make_plan(root):
    plan_path = root / "download_plan.json"
    if plan_path.exists():
        raise FileExistsError(f"Plan already exists: {plan_path}. Use run to resume it.")
    mix = json.loads((REPO / "experiments/qwen3-tiny/data_mix.json").read_text(encoding="utf-8"))
    sources, queues = [], []
    for source in mix["sources"]:
        name, repo_id, revision = source["name"], source["repo_id"], source["revision"]
        prefix, target = PREFIXES[name]
        url = f"https://huggingface.co/api/datasets/{repo_id}/tree/{revision}/{prefix}?limit=1000"
        catalog = []
        while url:
            entries, link = fetch_json(url)
            catalog.extend(e for e in entries if e["type"] == "file" and e["path"].endswith(".parquet"))
            match = re.search(r'<([^>]+)>;\s*rel="next"', link)
            url = match.group(1) if match else None
        if not catalog:
            raise RuntimeError(f"No parquet shards found for {name}")
        catalog.sort(key=lambda item: item["path"])
        random.Random(f'{mix["seed"]}:{name}').shuffle(catalog)
        selected, total = [], 0
        for entry in catalog:
            sha = entry.get("lfs", {}).get("oid", "")
            if not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise RuntimeError(f"No official SHA-256 for {entry['path']}")
            selected.append({
                "source": name, "repo_id": repo_id, "revision": revision,
                "remote_path": entry["path"], "bytes": entry["size"], "sha256": sha,
                "local_path": f"raw/{name}/{Path(entry['path']).name}",
                "url": f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{entry['path']}",
            })
            total += entry["size"]
            if target is not None and total >= target:
                break
        if target is not None and total < target:
            raise RuntimeError(f"Insufficient source bytes for {name}")
        queues.append(selected)
        sources.append({**source, "catalog_files": len(catalog), "selected_files": len(selected),
                        "selected_bytes": total, "requested_bytes": target})
        write_json(root / "metadata" / name / "catalog.json", catalog)
        # Dataset card is required provenance, not a trainable corpus file.
        with open_url(f"https://huggingface.co/datasets/{repo_id}/raw/{revision}/README.md") as response:
            (root / "metadata" / name / "README.md").write_bytes(response.read())
        print(f"{name}: {len(selected)} shards, {total / 1e9:.2f} GB", flush=True)
    # Interleave sources so that early downloads cover every component.
    files = [queue[index] for index in range(max(map(len, queues))) for queue in queues if index < len(queue)]
    plan = {"schema_version": 1, "created_at": now(), "seed": mix["seed"],
            "note": "Raw acquisition budget, not final token proportions or a cleaned training set.",
            "sources": sources, "total_bytes": sum(f["bytes"] for f in files), "files": files}
    write_json(plan_path, plan)
    print(f"Plan: {plan_path}; total {plan['total_bytes'] / 1e9:.2f} GB", flush=True)


def local_path(root, entry):
    path = (root / entry["local_path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Download destination escapes the data directory")
    return path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def receipt_path(path):
    return path.with_name(path.name + ".verified.json")


def is_verified(path, entry):
    try:
        receipt = json.loads(receipt_path(path).read_text(encoding="utf-8"))
        stat = path.stat()
        return (receipt["sha256"] == entry["sha256"] and stat.st_size == entry["bytes"]
                and stat.st_mtime_ns == receipt["mtime_ns"])
    except (OSError, ValueError, KeyError):
        return False


def transfer(entry, root, update, attempts=10):
    """Resume .part files; never rename a file to .parquet before hash verification."""
    path = local_path(root, entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    if is_verified(path, entry):
        update(entry, "verified", "Previously verified file unchanged")
        return
    for attempt in range(attempts):
        try:
            if path.exists():
                if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                    raise RuntimeError(f"Existing final file failed verification: {path}")
            else:
                offset = part.stat().st_size if part.exists() else 0
                if offset > entry["bytes"]:
                    raise RuntimeError(f"Partial file larger than expected: {part}")
                if offset < entry["bytes"]:
                    update(entry, "downloading", f"Attempt {attempt + 1}, resuming at {offset} bytes")
                    headers = {"Range": f"bytes={offset}-"} if offset else {}
                    # Fresh redirect avoids an expired signed CDN URL cached by an intermediary.
                    separator = "&" if "?" in entry["url"] else "?"
                    url = entry["url"] + separator + f"download=true&nonce={time.time_ns()}"
                    with open_url(url, headers) as response:
                        status = response.status
                        if status == 206:
                            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                            if not match or int(match[1]) != offset or int(match[3]) != entry["bytes"]:
                                raise RuntimeError("Unexpected Content-Range; refusing to append")
                        elif status == 200:
                            offset = 0  # Server ignored Range; restart safely rather than append.
                        else:
                            raise RuntimeError(f"Unexpected HTTP status {status}")
                        mode = "ab" if offset else "wb"
                        with part.open(mode) as output:
                            received = offset
                            while chunk := response.read(1024 * 1024):
                                if received + len(chunk) > entry["bytes"]:
                                    raise RuntimeError("Server sent more bytes than expected")
                                output.write(chunk)
                                received += len(chunk)
                    if part.stat().st_size != entry["bytes"]:
                        raise RuntimeError("Incomplete response; will resume")
                update(entry, "verifying", "Computing SHA-256 against official LFS checksum")
                actual = digest(part)
                if actual != entry["sha256"]:
                    bad = part.with_name(part.name + f".corrupt-{time.time_ns()}")
                    part.rename(bad)
                    raise RuntimeError(f"SHA-256 mismatch; quarantined as {bad.name}")
                os.replace(part, path)
            write_json(receipt_path(path), {
                "sha256": entry["sha256"], "bytes": entry["bytes"],
                "mtime_ns": path.stat().st_mtime_ns, "verified_at": now(),
                "repo_id": entry["repo_id"], "revision": entry["revision"], "remote_path": entry["remote_path"],
            })
            update(entry, "verified", "SHA-256 matches official checksum")
            return
        except Exception as exc:
            # Do not emit signed redirect URLs (which may contain temporary credentials).
            message = f"{type(exc).__name__}: {str(exc).split('https://')[0][:300]}"
            update(entry, "retrying" if attempt + 1 < attempts else "failed", message)
            if attempt + 1 == attempts:
                raise RuntimeError(f"Failed {entry['local_path']}: {message}") from None
            time.sleep(min(2 ** attempt, 30))


def snapshot(root, plan, states=None):
    rows = []
    for entry in plan["files"]:
        path = local_path(root, entry)
        part = path.with_name(path.name + ".part")
        candidate = path if path.exists() else part
        try:
            size = candidate.stat().st_size
        except FileNotFoundError:
            size = 0
        row = {"source": entry["source"], "file": entry["local_path"], "downloaded_bytes": size,
               "expected_bytes": entry["bytes"], "state": "queued"}
        row.update((states or {}).get(entry["local_path"], {}))
        if is_verified(path, entry):
            row["state"] = "verified"
        rows.append(row)
    return {"updated_at": now(), "pid": os.getpid(), "planned_bytes": plan["total_bytes"],
            "downloaded_bytes": sum(r["downloaded_bytes"] for r in rows),
            "verified_bytes": sum(r["expected_bytes"] for r in rows if r["state"] == "verified"),
            "verified_files": sum(r["state"] == "verified" for r in rows), "total_files": len(rows), "files": rows}


def run(root, workers, reserve_gb):
    plan = json.loads((root / "download_plan.json").read_text(encoding="utf-8"))
    # OS lock is released on process exit; no stale PID file can block a restart.
    with (root / "download.lock").open("a+b") as lock:
        lock.seek(0)
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        states, mutex, stop = {}, threading.Lock(), threading.Event()
        initial = snapshot(root, plan)
        remaining = plan["total_bytes"] - initial["downloaded_bytes"]
        if shutil.disk_usage(root).free < remaining + reserve_gb * 1e9:
            raise RuntimeError("Insufficient space for the plan plus the requested free-space reserve")

        def update(entry, state, message):
            with mutex:
                states[entry["local_path"]] = {"state": state, "message": message, "updated_at": now()}
                print(f"{now()} [{state}] {entry['local_path']}: {message}", flush=True)

        def save_progress():
            with mutex:
                result = snapshot(root, plan, states)
                write_json(root / "progress.json", result)
            return result

        def monitor():
            while not stop.wait(5):
                save_progress()

        save_progress()
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        failures = []
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(transfer, entry, root, update) for entry in plan["files"]]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        failures.append(str(exc))
        finally:
            stop.set()
            thread.join()
            result = save_progress()
            result["finished_at"] = now()
            result["status"] = "complete" if result["verified_files"] == result["total_files"] else "incomplete"
            result["errors"] = failures
            write_json(root / "progress.json", result)
        if failures:
            raise RuntimeError(f"{len(failures)} downloads failed; rerun run to resume. See progress.json.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run", "status"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reserve-gb", type=float, default=100)
    args = parser.parse_args()
    if args.workers < 1 or args.reserve_gb < 0:
        parser.error("workers must be positive and reserve-gb nonnegative")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "plan":
        make_plan(root)
    elif args.command == "run":
        run(root, args.workers, args.reserve_gb)
    else:
        plan = json.loads((root / "download_plan.json").read_text(encoding="utf-8"))
        print(json.dumps(snapshot(root, plan), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
