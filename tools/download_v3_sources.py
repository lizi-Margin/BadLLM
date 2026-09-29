"""Pinned, resumable, SHA256 verified acquisition for the 3B mixture."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import re
import time
import download_pretrain as dl
ROOT = Path(__file__).resolve().parents[1]

def catalog(repo, revision, prefix):
    url = f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/{prefix}?recursive=true&limit=1000'
    rows = []
    while url:
        data, link = dl.fetch_json(url)
        rows.extend(x for x in data if x['type'] == 'file' and x['path'].endswith('.parquet'))
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = match[1] if match else None
    return rows

def plan(extra=0):
    root = ROOT / 'data/pretrain'
    mix = json.loads((ROOT / 'experiments/qwen3-tiny/data_mix_v3.json').read_text())
    old = json.loads((root / 'download_plan.json').read_text())
    old_sources = {s['name']: s for s in old['sources']}
    dest = root / 'v3_download_plan.json'
    prior = json.loads(dest.read_text()) if dest.exists() else None
    revisions = {s['repo_id']: s['revision'] for s in prior['sources'] if 'repo_id' in s} if prior else {}
    entries = []
    for s in mix['sources']:
        if 'local_source' in s:
            o = old_sources[s['local_source']]
            s.update({k: o[k] for k in ['repo_id','revision','license_note']})
            for e in old['files']:
                if e['source'] == s['local_source']:
                    entries.append({**e, 'source': s['name'], 'language': ''})
            continue
        repo = s['repo_id']
        if repo not in revisions:
            revisions[repo] = dl.fetch_json(f'https://huggingface.co/api/datasets/{repo}')[0]['sha']
        rev = s['revision'] = revisions[repo]
        md = root / 'metadata_v3' / s['name']; md.mkdir(parents=True,exist_ok=True)
        with dl.open_url(f'https://huggingface.co/datasets/{repo}/raw/{rev}/README.md') as r:
            (md/'README.md').write_bytes(r.read())
        for lang in s.get('languages', {'': 1}):
            prefix = s['remote_prefix'] + ('/'+lang if lang else '')
            rows = catalog(repo, rev, prefix)
            dl.write_json(md / f'catalog_{lang or "all"}.json', rows)
            existing = [x for x in rows if (root/'raw_v3'/s['name']/Path(x['path']).name).exists()]
            remaining = [x for x in rows if x not in existing]
            random.Random('42:'+s['name']+lang).shuffle(remaining)
            n = (4 if s['name'] == 'ultradata_math_l2' else 2 if not lang else 1) + extra
            selected = (existing + remaining)[:max(n,len(existing))]
            for x in selected:
                sha = x.get('lfs',{}).get('oid')
                if not sha or not re.fullmatch('[a-f0-9]{64}',sha): raise ValueError('Missing official SHA256')
                entries.append(dict(source=s['name'],language=lang,repo_id=repo,revision=rev,
                    remote_path=x['path'],bytes=x['size'],sha256=sha,
                    local_path=f'raw_v3/{s["name"]}/{Path(x["path"]).name}',
                    url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{x["path"]}'))
    result={'sources':mix['sources'],'files':entries,'total_bytes':sum(x['bytes'] for x in entries),'mix':mix}
    dl.write_json(dest,result)
    print(json.dumps({'plan':str(dest),'files':len(entries),'bytes':result['total_bytes']}),flush=True)
    return result

def run(workers,extra):
    p=plan(extra); root=ROOT/'data/pretrain'
    def transfer(e):
        path=dl.local_path(root,e)
        if path.exists() and not dl.is_verified(path,e):
            if path.stat().st_size != e['bytes'] or dl.digest(path) != e['sha256']:
                path.rename(path.with_name(path.name+f'.corrupt-{time.time_ns()}'))
        dl.transfer(e,root,lambda e,state,msg:print(f"{e['source']} {Path(e['remote_path']).name}: {state} {msg}",flush=True))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(transfer,p['files']))
    dl.write_json(root/'v3_download_verified.json',dict(status='complete',files=len(p['files']),bytes=p['total_bytes'],time=dl.now()))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--extra',type=int,default=0)
    a=ap.parse_args(); run(a.workers,a.extra)
