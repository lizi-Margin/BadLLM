"""Append official Ultra-FineWeb-ZH shards for the unseen-document replay quota."""
import json, random, re
from pathlib import Path
import download_pretrain as dl
from download_v3_sources import catalog

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/pretrain'
PLAN_PATH = DATA / 'v5_download_plan.json'

def extend(count=4):
    plan = json.loads(PLAN_PATH.read_text(encoding='utf-8'))
    old = json.loads((DATA / 'v3_download_plan.json').read_text(encoding='utf-8'))
    src = next(s for s in old['sources'] if s.get('local_source') == 'ultrafineweb_zh')
    repo, rev, prefix = src['repo_id'], src['revision'], 'data/ultrafineweb_zh'
    rows = catalog(repo, rev, prefix)
    used = {e['remote_path'] for e in plan['files'] if e['source'] == 'ultrafineweb_zh_fresh'}
    remaining = [r for r in rows if r['path'] not in used]
    random.Random('v5-extra-42').shuffle(remaining)
    added=[]
    for x in remaining[:count]:
        sha=x.get('lfs',{}).get('oid')
        if not sha or not re.fullmatch(r'[a-f0-9]{64}',sha): raise ValueError(x['path'])
        e=dict(source='ultrafineweb_zh_fresh',language='',repo_id=repo,revision=rev,
            remote_path=x['path'],bytes=x['size'],sha256=sha,
            local_path=f'raw_v5/ultrafineweb_zh_fresh/{Path(x["path"]).name}',
            url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{x["path"]}')
        plan['files'].append(e); added.append(e)
    plan['total_bytes']=sum(e['bytes'] for e in plan['files'])
    dl.write_json(PLAN_PATH,plan)
    print(json.dumps({'added':[(e['remote_path'],e['bytes']) for e in added]},ensure_ascii=False),flush=True)
    for e in added:
        dl.transfer(e,DATA,lambda _e,state,msg:print(f"{Path(e['remote_path']).name}: {state} {msg}",flush=True))

if __name__=='__main__': extend()
