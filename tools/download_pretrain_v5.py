"""Create a pinned, resumable source plan for the reasoning-focused v5 corpus."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import re
import time
import download_pretrain as dl
from download_v3_sources import catalog

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/pretrain'
MIX_PATH = ROOT / 'experiments/qwen3-tiny/data_mix_v5_reasoning.json'
PLAN_PATH = DATA / 'v5_download_plan.json'

def make_plan(extra: int = 0):
    mix = json.loads(MIX_PATH.read_text(encoding='utf-8'))
    previous = json.loads((DATA / 'v3_download_plan.json').read_text(encoding='utf-8'))
    previous_sources = {s['name']: s for s in previous['sources']}
    pinned = {}
    for s in previous['sources']:
        if 'repo_id' in s:
            pinned[s['repo_id'].lower()] = s['revision']
    entries = []
    for s in mix['sources']:
        name = s['name']
        if 'local_source' in s:
            source = s['local_source']
            old_source = previous_sources.get(source) or next((x for x in previous['sources'] if x.get('local_source') == source), None)
            if source == 'ultrafineweb_zh':
                old_source = next((x for x in previous['sources'] if x['name'] == 'ultrafineweb_zh_replay'), old_source)
            old_key = old_source['name'] if old_source else source
            old_entries = [e for e in previous['files'] if e['source'] == old_key]
            if not old_entries or not old_source:
                raise RuntimeError(f'No verified catalog for local source {source}')
            s.update({k: old_source[k] for k in ('repo_id','revision','license_note') if k in old_source})
            for e in old_entries:
                entries.append({**e, 'source': name, 'language': ''})
            continue
        # Reuse already verified v3 shards for the same pinned source/config.
        reusable = [e for e in previous['files'] if e['source'] == name]
        if reusable:
            for e in reusable:
                entries.append({**e, 'source': name})
            continue
        repo = s['repo_id']
        revision = pinned.get(repo.lower())
        if not revision:
            revision = dl.fetch_json(f'https://huggingface.co/api/datasets/{repo}')[0]['sha']
        s['revision'] = revision
        md = DATA / 'metadata_v5' / name
        md.mkdir(parents=True, exist_ok=True)
        if not (md / 'README.md').exists():
            with dl.open_url(f'https://huggingface.co/datasets/{repo}/raw/{revision}/README.md') as r:
                (md / 'README.md').write_bytes(r.read())
        for lang in s.get('languages', {'': 1.0}):
            prefix = s['remote_prefix'] + ('/' + lang if lang else '')
            rows = catalog(repo, revision, prefix)
            dl.write_json(md / f'catalog_{lang or "all"}.json', rows)
            local_dir = DATA / 'raw_v5' / name
            existing = [x for x in rows if (local_dir / Path(x['path']).name).exists()]
            remaining = [x for x in rows if x not in existing]
            random.Random('v5:42:' + name + lang).shuffle(remaining)
            base = 1
            selected = (existing + remaining)[:max(base + extra, len(existing))]
            for x in selected:
                sha = x.get('lfs', {}).get('oid')
                if not sha or not re.fullmatch(r'[a-f0-9]{64}', sha):
                    raise ValueError(f'Missing official SHA-256: {name}/{x["path"]}')
                entries.append(dict(source=name, language=lang, repo_id=repo, revision=revision,
                    remote_path=x['path'], bytes=x['size'], sha256=sha,
                    local_path=f'raw_v5/{name}/{Path(x["path"]).name}',
                    url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{x["path"]}'))
    plan = dict(sources=mix['sources'], files=entries, total_bytes=sum(x['bytes'] for x in entries), mix=mix)
    dl.write_json(PLAN_PATH, plan)
    print(json.dumps(dict(plan=str(PLAN_PATH), files=len(entries), bytes=plan['total_bytes']), ensure_ascii=False))
    return plan

def download(workers: int, extra: int):
    p = make_plan(extra)
    def transfer(e):
        path = dl.local_path(DATA, e)
        if path.exists() and not dl.is_verified(path, e):
            path.rename(path.with_name(path.name + f'.corrupt-{time.time_ns()}'))
        dl.transfer(e, DATA, lambda _e, state, msg: print(f"{e['source']} {Path(e['remote_path']).name}: {state} {msg}", flush=True))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(transfer, p['files']))
    dl.write_json(DATA / 'v5_download_verified.json', dict(status='complete', files=len(p['files']), bytes=p['total_bytes'], time=dl.now()))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--extra', type=int, default=4)
    args = ap.parse_args()
    download(args.workers, args.extra)
