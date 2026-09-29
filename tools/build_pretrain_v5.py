"""Build the exact 3B mixture with resumable document transactions and native tokenization.

Windows: .venv-data/Scripts/python.exe tools/build_pretrain_v3.py --output data/pretrain/prepared/pt-3b-v4
Source documents and pretokenized uint32 streams are both exported. Each source/language
has its own quota, including EOS. A final document may be token-truncated to meet an
exact quota; full text remains in the audit record and used_tokens records the boundary.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import time
import unicodedata
os.environ.setdefault('RAYON_NUM_THREADS','16')
os.environ['TOKENIZERS_PARALLELISM']='true'
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from download_pretrain import is_verified, write_json, digest as file_digest

ROOT=Path(__file__).resolve().parents[1]
VERSION='quota-v5.0'

def clean(text):
    if not isinstance(text,str): return ''
    text=unicodedata.normalize('NFC',text).replace('\r\n','\n').replace('\r','\n')
    # Preserve code indentation and all meaningful internal whitespace.
    text='\n'.join(x.rstrip() for x in text.split('\n'))
    text=re.sub(r'\n{4,}','\n\n\n',text).strip()
    if len(text)<80 or len(text)>200000 or '\x00' in text or text.count('\ufffd')>2: return ''
    if sum(c.isprintable() or c in '\n\t' for c in text)/len(text)<.97: return ''
    return text

def sha(text): return hashlib.sha256(text.encode()).hexdigest()

def open_db(out):
    db=sqlite3.connect(out/'documents.sqlite3')
    db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=NORMAL')
    db.execute('PRAGMA cache_size=-131072'); db.execute('PRAGMA temp_store=FILE')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT);
    CREATE TABLE IF NOT EXISTS docs (hash TEXT PRIMARY KEY, split TEXT, source TEXT, language TEXT, tokens INTEGER, shuffle TEXT, ids BLOB, record TEXT);
    CREATE TABLE IF NOT EXISTS cursor (task TEXT PRIMARY KEY, file INTEGER, batch INTEGER, train INTEGER, validation INTEGER, documents INTEGER, filtered TEXT);
    CREATE TABLE IF NOT EXISTS groups (key TEXT PRIMARY KEY, split TEXT);
    ''')
    return db

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--plan',type=Path,default=ROOT/'data/pretrain/v5_download_plan.json'); ap.add_argument('--exclude-db',type=Path,action='append',default=[]); ap.add_argument('--batch-size',type=int,default=512); ap.add_argument('--scale',type=float,default=1.0); ap.add_argument('--export-only',action='store_true'); args=ap.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    # OS lock prevents two builders from corrupting progress or exports.
    lock=(out/'build.lock').open('a+b'); lock.write(b'0'); lock.flush(); lock.seek(0)
    if os.name=='nt':
        import msvcrt
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads(args.plan.read_text(encoding='utf-8')); mix=plan['mix']
    tokenizer_path=ROOT/mix['tokenizer']
    tokenizer=Tokenizer.from_file(str(tokenizer_path/'tokenizer.json'))
    eos=json.loads((tokenizer_path/'tokenizer_config.json').read_text(encoding='utf-8'))['eos_token']
    if isinstance(eos,dict): eos=eos['content']
    eos_id=tokenizer.token_to_id(eos)
    if eos_id is None: raise RuntimeError('EOS missing')
    special={tokenizer.token_to_id(x['content']) for x in json.loads((tokenizer_path/'tokenizer.json').read_text(encoding='utf-8'))['added_tokens'] if x['special']}
    db=open_db(out)
    # Keep exact hashes in RAM: per-document SQLite SELECTs dominate tokenization time
    # once the source contains millions of short documents.
    seen_hashes={r[0] for r in db.execute('SELECT hash FROM docs')}
    seen_groups={r[0]:r[1] for r in db.execute('SELECT key,split FROM groups')}
    signature=sha(json.dumps(dict(version=VERSION,mix=mix,scale=args.scale,batch=args.batch_size,tokenizer=file_digest(tokenizer_path/'tokenizer.json')),sort_keys=True))
    previous=db.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
    if previous and previous[0]!=signature: raise RuntimeError('Build configuration changed; use a new output directory')
    db.execute("INSERT OR IGNORE INTO meta VALUES ('signature',?)",(signature,)); db.commit()
    # Exclude every prior train/validation document by normalized-text SHA256.
    exclude_paths=args.exclude_db or [
        ROOT/'data/pretrain/prepared/full-50gb-v3/seen_hashes.sqlite3',
        ROOT/'data/pretrain/prepared/pt-3b-v4/documents.sqlite3']
    excluded_hashes=set(); exclusion_sources=[]
    for path in exclude_paths:
        path=path.resolve(); conn=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
        tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'seen' in tables: rows=(r[0] for r in conn.execute('SELECT hash FROM seen'))
        elif 'docs' in tables: rows=(r[0] for r in conn.execute('SELECT hash FROM docs'))
        else:
            conn.close(); raise RuntimeError(f'No document hash table in {path}')
        before=len(excluded_hashes); excluded_hashes.update(rows)
        exclusion_sources.append(dict(path=str(path),new_hashes=len(excluded_hashes)-before)); conn.close()
    print(json.dumps(dict(excluded_documents=len(excluded_hashes),databases=exclusion_sources)),flush=True)
    progress={}; started=time.monotonic(); previous_report=0
    for s in mix['sources']:
        for language,fraction in s.get('languages',{'':1.0}).items():
            name=s['name']; task=name+(':'+language if language else '')
            target={'train':round(s['tokens']*fraction*args.scale),'validation':max(64,round(s['tokens']*fraction*args.scale*mix['validation_fraction']))}
            row=db.execute('SELECT file,batch,train,validation,documents,filtered FROM cursor WHERE task=?',(task,)).fetchone()
            fi,bi,tr,va,nd,filtered=row or (0,0,0,0,0,'{}'); used={'train':tr,'validation':va}; filters=Counter(json.loads(filtered))
            files=[e for e in plan['files'] if e['source']==name and e.get('language','')==language]
            random.Random('42:'+task).shuffle(files)
            # New files may only be appended to an existing task file order.
            saved=db.execute('SELECT value FROM meta WHERE key=?',('files:'+task,)).fetchone()
            if saved:
                ordered=json.loads(saved[0]); bypath={e['local_path']:e for e in files}
                if any(x not in bypath for x in ordered): raise RuntimeError('Source files removed')
                files=[bypath.pop(x) for x in ordered]+list(bypath.values())
            db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('files:'+task,json.dumps([e['local_path'] for e in files])));db.commit()
            if not args.export_only and not all(used[k]==target[k] for k in target):
                for file_index,e in enumerate(files):
                    if file_index<fi: continue
                    path=ROOT/'data/pretrain'/e['local_path']
                    if not is_verified(path,e): raise RuntimeError(f'Unverified shard: {path}')
                    pf=pq.ParquetFile(path); column=s['text_column']
                    if column not in pf.schema_arrow.names: raise RuntimeError(f'Missing {column}: {path}')
                    cols=[column]+[c for c in ['uid','uuid','id','url','title','repo_name','relative_path','raw_content'] if c in pf.schema_arrow.names and c!=column]
                    for batch_index,batch in enumerate(pf.iter_batches(batch_size=args.batch_size,columns=cols)):
                        if file_index==fi and batch_index<bi: continue
                        candidates=[]
                        for index,r in enumerate(batch.to_pylist()):
                            text=clean(r[column])
                            if not text: filters['quality']+=1;continue
                            h=sha(text)
                            if h in excluded_hashes:
                                filters['previous_dataset_duplicate']+=1;continue
                            # Related synthetic variants stay in the same split.
                            group=h
                            if name.startswith('ultrafineweb_l3') and r.get('uid'): group=sha('ultrafineweb:'+str(r['uid']))
                            elif name.startswith('ultradata_code_l3') and r.get('raw_content'):
                                raw=clean(r['raw_content']); group=sha(raw) if raw else h
                            validation_ratio=target['validation']/max(1,target['train'])
                            validation_cutoff=int(validation_ratio*(1<<32))
                            sp='validation' if int(group[:8],16)<validation_cutoff else 'train'
                            if group in seen_groups: sp=seen_groups[group]
                            if used[sp]>=target[sp]:continue
                            if h in seen_hashes: filters['duplicate']+=1;continue
                            candidates.append((r,text,h,group,sp,index))
                        encodings=tokenizer.encode_batch([x[1] for x in candidates],add_special_tokens=False)
                        for (r,text,h,group,sp,index),enc in zip(candidates,encodings):
                            if used[sp]>=target[sp]:continue
                            ids=enc.ids
                            if len(ids)<32 or special.intersection(ids):filters['short_or_special']+=1;continue
                            total=len(ids)+1; take=min(total,target[sp]-used[sp])
                            if take<2: raise RuntimeError('Invalid tiny quota remainder')
                            # Avoid a one-token remainder by trimming this final-but-one document.
                            if target[sp]-used[sp]-take==1: take-=1
                            ids=ids[:take-1]+[eos_id]
                            metadata={k:v for k,v in r.items() if k not in [column,'raw_content']}
                            rec=dict(text=text,source=name,language=language,document_hash=h,group_hash=group,
                                tokens=take,original_tokens=total,truncated=take!=total,
                                origin=dict(repo_id=e['repo_id'],revision=e['revision'],remote_path=e['remote_path'],raw_sha256=e['sha256'],row=batch_index*args.batch_size+index,metadata=metadata))
                            inserted=db.execute('INSERT OR IGNORE INTO docs VALUES (?,?,?,?,?,?,?,?)',(h,sp,name,language,take,sha('shuffle42:'+h),np.asarray(ids,dtype='<u4').tobytes(),json.dumps(rec,ensure_ascii=False,separators=(',',':')))).rowcount
                            if inserted:
                                seen_hashes.add(h)
                                seen_groups.setdefault(group,sp)
                                db.execute('INSERT OR IGNORE INTO groups VALUES (?,?)',(group,sp));used[sp]+=take;nd+=1
                            else: filters['duplicate']+=1
                        db.execute('INSERT OR REPLACE INTO cursor VALUES (?,?,?,?,?,?,?)',(task,file_index,batch_index+1,used['train'],used['validation'],nd,json.dumps(dict(filters))));db.commit()
                        progress[task]=dict(actual=used.copy(),target=target,documents=nd,filtered=dict(filters))
                        if time.monotonic()-previous_report>10:
                            write_json(out/'progress.json',dict(status='building',pid=os.getpid(),time=time.time(),current_task=task,tasks=progress))
                            print(json.dumps(dict(task=task,actual=used,target=target,documents=nd,elapsed=round(time.monotonic()-started))),flush=True);previous_report=time.monotonic()
                        if all(used[k]==target[k] for k in target):break
                    pf.close()
                    if all(used[k]==target[k] for k in target):break
            progress[task]=dict(actual=used.copy(),target=target,documents=nd,filtered=dict(filters))
            if not all(used[k]==target[k] for k in target):
                write_json(out/'progress.json',dict(status='source_exhausted',pid=os.getpid(),task=task,tasks=progress));raise RuntimeError(f'Source exhausted {task}: {used}/{target}; acquire more verified shards and resume')
    write_json(out/'progress.json',dict(status='exporting',pid=os.getpid(),tasks=progress,time=time.time()))
    print('All token quotas reached; exporting shuffled documents and packed token streams',flush=True)
    db.execute('CREATE INDEX IF NOT EXISTS export_order ON docs(split,shuffle)');db.commit()
    outputs={}
    for sp in ['train','validation']:
        binary=out/f'{sp}.bin'; jsonl=out/f'{sp}.jsonl'; hb=hashlib.sha256(); hj=hashlib.sha256(); n=0; nt=0
        with binary.with_suffix('.bin.tmp').open('wb') as bf,jsonl.with_suffix('.jsonl.tmp').open('wb') as jf:
            for ids,record,ntok in db.execute('SELECT ids,record,tokens FROM docs WHERE split=? ORDER BY shuffle',(sp,)):
                b=(record+'\n').encode();bf.write(ids);jf.write(b);hb.update(ids);hj.update(b);n+=1;nt+=ntok
                if n%50000==0:print(f'export {sp}: {n} documents, {nt} tokens',flush=True)
        os.replace(binary.with_suffix('.bin.tmp'),binary);os.replace(jsonl.with_suffix('.jsonl.tmp'),jsonl)
        outputs[sp]=dict(tokens=nt,documents=n,binary=dict(file=binary.name,bytes=binary.stat().st_size,sha256=hb.hexdigest()),jsonl=dict(file=jsonl.name,bytes=jsonl.stat().st_size,sha256=hj.hexdigest()),block_size=1024,blocks=(nt+1023)//1024,padding_tokens=(-nt)%1024)
    manifest=dict(schema_version=4,status='complete',builder_version=VERSION,signature=signature,mix=mix,scale=args.scale,tokenizer_sha256=file_digest(tokenizer_path/'tokenizer.json'),eos_token_id=eos_id,pad_token_id=tokenizer.token_to_id('<|endoftext|>'),outputs=outputs,tasks=progress,
        dedup='Global NFC text SHA256; related synthetic UID/raw-content groups split together; all exact document hashes from prior 50GB-v3 and 3B-v4 train and validation exports are excluded',
        quota='Exact tokens including one EOS per document; at most one truncated document per source/language/split',
        packing='Document shuffle then consecutive uint32 little-endian token stream, no overlap or dropped tails; loader masks final padding',
        exclusions=exclusion_sources,
        limitations=['Exact and known-family dedup only; unknown near-duplicates and benchmark contamination not screened.','The 5% Ultra-FineWeb-ZH share is unseen source text, preserving source distribution without replaying prior documents.'],created_at=time.time())
    write_json(out/'manifest.json',manifest)
    write_json(out/'dataset_info.json',{f'v3_{sp}':{'file_name':sp+'.jsonl','columns':{'prompt':'text'}} for sp in outputs})
    write_json(out/'progress.json',dict(status='complete',pid=os.getpid(),time=time.time(),outputs=outputs,tasks=progress))
    db.close();print(json.dumps(outputs,indent=2),flush=True)

if __name__=='__main__':main()

