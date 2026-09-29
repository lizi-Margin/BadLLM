"""Train directly from the verified prepacked v3 dataset, without retokenization."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

class PackedDataset:
    def __init__(self, directory, split='train', block_size=1024):
        directory = Path(directory)
        self.manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        if self.manifest['status'] != 'complete': raise ValueError('Dataset not complete')
        info = self.manifest['outputs'][split]
        self.path = directory/info['binary']['file']
        self.count = info['tokens']
        self.block_size = block_size
        self.pad_id = self.manifest['pad_token_id']
        if self.path.stat().st_size != self.count * 4: raise ValueError('Binary length mismatch')
        self.tokens = np.memmap(self.path, mode='r', dtype='<u4')

    def __len__(self): return math.ceil(self.count/self.block_size)

    def __getitem__(self, index):
        if index < 0 or index >= len(self): raise IndexError(index)
        start = index*self.block_size
        part = np.asarray(self.tokens[start:start+self.block_size], dtype=np.int64)
        ids = np.full(self.block_size, self.pad_id, dtype=np.int64)
        mask = np.zeros(self.block_size, dtype=np.int64)
        labels = np.full(self.block_size, -100, dtype=np.int64)
        ids[:len(part)] = part; labels[:len(part)] = part; mask[:len(part)] = 1
        return dict(input_ids=ids, attention_mask=mask, labels=labels)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,default=ROOT/'experiments/qwen3-tiny/pt-3b-v4.json')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--resume',type=str)
    ap.add_argument('--max-steps',type=int)
    ap.add_argument('--output',type=Path)
    a=ap.parse_args()
    import torch
    from torch.utils.data import Subset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, default_data_collator, set_seed
    cfg=json.loads(a.config.read_text(encoding='utf-8'))
    data=ROOT/cfg['dataset']; model_path=ROOT/cfg['model']
    train=PackedDataset(data); val=PackedDataset(data,'validation')
    tokenizer_hash=hashlib.sha256((model_path/'tokenizer.json').read_bytes()).hexdigest()
    if tokenizer_hash != train.manifest['tokenizer_sha256']: raise RuntimeError('Model tokenizer differs from dataset')
    set_seed(cfg['seed'])
    torch.set_num_threads(8)
    model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa')
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    param_count=sum(x.numel() for x in model.parameters())
    before=model.model.layers[0].mlp.down_proj.weight.detach().float().clone()
    output=a.output or ROOT/(cfg['output']+('-smoke' if a.smoke else ''))
    steps=a.max_steps or (16 if a.smoke else -1)
    if a.smoke:
        # Spread evaluation blocks through the shuffled holdout; training reads the real full file.
        val=Subset(val,np.linspace(0,len(val)-1,min(32,len(val)),dtype=int).tolist())
    class FiniteCallback(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            for k,v in (logs or {}).items():
                if isinstance(v,(float,int)) and not math.isfinite(v): raise RuntimeError(f'Nonfinite {k}: {v}')
    ta=TrainingArguments(output_dir=str(output),per_device_train_batch_size=8 if a.smoke else cfg['batch_size'],
        per_device_eval_batch_size=4,gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        learning_rate=cfg['learning_rate'],lr_scheduler_type='cosine',warmup_ratio=.02,
        num_train_epochs=1,max_steps=steps,bf16=True,tf32=True,gradient_checkpointing=False,
        optim='adamw_torch_fused',logging_steps=1 if a.smoke else 100,
        save_strategy='steps',save_steps=8 if a.smoke else 5000,save_total_limit=2,
        report_to='none',seed=cfg['seed'],dataloader_num_workers=0 if a.smoke else 2,
        remove_unused_columns=False,eval_strategy='no',logging_nan_inf_filter=False)
    trainer=Trainer(model=model,args=ta,train_dataset=train,eval_dataset=val,
        processing_class=tokenizer,data_collator=default_data_collator,callbacks=[FiniteCallback()])
    pre_eval=trainer.evaluate() if a.smoke and not a.resume else None
    result=trainer.train(resume_from_checkpoint=a.resume)
    post_eval=trainer.evaluate() if a.smoke else None
    delta=float((model.model.layers[0].mlp.down_proj.weight.detach().float().cpu()-before.cpu()).abs().max())
    if delta<=0: raise RuntimeError('Selected model parameter did not update')
    trainer.save_model(); trainer.save_state(); tokenizer.save_pretrained(output)
    trainer.save_metrics('train',result.metrics)
    if post_eval: trainer.save_metrics('eval',post_eval)
    evidence=dict(status='passed',model=str(model_path),parameter_count=param_count,dataset=str(data),
        dataset_signature=train.manifest['signature'],train_tokens=train.count,train_blocks=len(train),
        max_parameter_delta=delta,global_step=trainer.state.global_step,pre_eval=pre_eval,post_eval=post_eval,
        metrics=result.metrics,log_history=trainer.state.log_history,resumed_from=a.resume,
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__)
    (output/('resume_test.json' if a.resume else 'usability_test.json')).write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in evidence.items() if k!='log_history'},indent=2),flush=True)

if __name__=='__main__':main()
