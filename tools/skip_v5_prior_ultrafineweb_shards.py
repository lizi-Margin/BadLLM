"""Skip remaining old Ultra-FineWeb shards after the old corpus hash audit."""
import json, sqlite3, time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'data/pretrain/prepared/pt-3b-v5-reasoning'
plan=json.loads((root/'data/pretrain/v5_download_plan.json').read_text(encoding='utf-8'))
old=[e for e in plan['files'] if e['source']=='ultrafineweb_zh_fresh' and e['local_path'].startswith('raw/')]
conn=sqlite3.connect(out/'documents.sqlite3')
row=conn.execute("SELECT file,batch,train,validation,documents FROM cursor WHERE task='ultrafineweb_zh_fresh'").fetchone()
if row is None: raise RuntimeError('Missing replay cursor')
if row[2] or row[3]: raise RuntimeError('Replay already has accepted rows; do not skip shards')
conn.execute("UPDATE cursor SET file=?,batch=0 WHERE task='ultrafineweb_zh_fresh'",(len(old),))
conn.execute("INSERT OR REPLACE INTO meta VALUES ('skipped_old_replay_shards',?)",(json.dumps(dict(count=len(old),prior_hash_dbs=['full-50gb-v3/seen_hashes.sqlite3','pt-3b-v4/documents.sqlite3'],reason='All earlier scanned records matched prior document hashes; remaining source shards belong to the same pinned files already used by the previous corpus',updated_at=time.time())),))
conn.commit();print(json.dumps(dict(previous_cursor=row,new_cursor=len(old),old_shards=len(old))))
conn.close()
