"""Independent full-export audit plus deterministic tokenizer spot checks."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
os.environ.setdefault('RAYON_NUM_THREADS','8')
import numpy as np
from tokenizers import Tokenizer
from train_pretrain_v3 import PackedDataset

ROOT=Path(__file__).resolve().parents[1]

def audit(directory):
    directory=Path(directory)
    m=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    assert m['status']=='complete'
    db=sqlite3.connect((directory/'documents.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
    tokenizer=Tokenizer.from_file(str(ROOT/m['mix']['tokenizer']/'tokenizer.json'))
    assert hashlib.sha256((ROOT/m['mix']['tokenizer']/'tokenizer.json').read_bytes()).hexdigest()==m['tokenizer_sha256']
    integrity=db.execute('PRAGMA integrity_check').fetchone()[0]; assert integrity=='ok'
    assert db.execute('SELECT count(*) FROM (SELECT group_hash FROM (SELECT json_extract(record,\'$.group_hash\') AS group_hash,split FROM docs) GROUP BY group_hash HAVING count(DISTINCT split)>1)').fetchone()[0]==0
    report=dict(status='passed',signature=m['signature'],sqlite_integrity=integrity,outputs={},sources={},tokenizer_checks=0,created_at=time.time())
    for split in ['train','validation']:
        ds=PackedDataset(directory,split); expected=m['outputs'][split]; binary=ds.tokens
        counts=Counter(); langs=Counter(); n=0; offset=0; bh=hashlib.sha256(); jh=hashlib.sha256(); checked=0
        with (directory/(split+'.jsonl')).open('rb') as stream:
            for h,source,language,nt,blob,record in db.execute('SELECT hash,source,language,tokens,ids,record FROM docs WHERE split=? ORDER BY shuffle',(split,)):
                line=stream.readline(); assert line==(record+'\n').encode('utf-8')
                row=json.loads(line); assert row['tokens']==nt and row['document_hash']==h
                assert hashlib.sha256(row['text'].encode()).hexdigest()==h
                ids=np.frombuffer(blob,dtype='<u4');assert len(ids)==nt and ids[-1]==m['eos_token_id']
                assert (ids<151936).all()
                assert np.array_equal(binary[offset:offset+nt],ids)
                if split=='validation' or n%1000==0 or row['truncated']:
                    enc=tokenizer.encode(row['text'],add_special_tokens=False).ids
                    assert row['original_tokens']==len(enc)+1
                    assert enc[:nt-1]+[m['eos_token_id']]==ids.tolist()
                    checked+=1
                bh.update(blob);jh.update(line);counts[source]+=nt;langs[(source,language)]+=nt;offset+=nt;n+=1
            assert not stream.read(1)
        assert offset==expected['tokens'] and offset==len(binary) and n==expected['documents']
        assert bh.hexdigest()==expected['binary']['sha256']
        assert jh.hexdigest()==expected['jsonl']['sha256']
        last=ds[len(ds)-1];tail=offset-(len(ds)-1)*1024
        assert int(last['attention_mask'].sum())==tail
        assert (last['labels'][tail:]==-100).all()
        assert len(ds)*1024-int((last['labels']==-100).sum())==offset
        for source in m['mix']['sources']:
            target=round(source['tokens']*m['scale']) if split=='train' else sum(x['target'][split] for k,x in m['tasks'].items() if k.split(':')[0]==source['name'])
            assert counts[source['name']]==target,(source['name'],counts[source['name']],target)
        for task,stats in m['tasks'].items():
            name,_,language=task.partition(':')
            assert langs[(name,language)]==stats['target'][split]
        report['outputs'][split]=dict(tokens=offset,documents=n,binary_sha256=bh.hexdigest(),jsonl_sha256=jh.hexdigest(),padding_tokens=1024-tail,tokenizer_checks=checked)
        report['sources'][split]=dict(counts);report['tokenizer_checks']+=checked
        print(json.dumps(dict(split=split,tokens=offset,documents=n,tokenizer_checks=checked)),flush=True)
    assert report['outputs']['train']['tokens']==round(3_000_000_000*m['scale'])
    report['global_unique_documents']=db.execute('SELECT count(*) FROM docs').fetchone()[0]
    db.close()
    (directory/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();print(json.dumps(audit(a.directory),indent=2))
