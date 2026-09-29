from pathlib import Path
p=Path('/mnt/d/BadLLM/wsl/vendor/LLaMA-Factory/src/llamafactory/model/loader.py')
s=p.read_text()
needle='    if is_trainable and getattr(model, "hf_device_map", None) is not None:\n'
insert='''    if is_trainable:
        # This is a single-GPU run; never let checkpoint metadata mark it as
        # model-parallel and make Trainer fall back to DataParallel.
        model.is_parallelizable = False
        model.model_parallel = False

'''
if needle not in s: raise SystemExit('missing')
p.write_text(s.replace(needle,insert+needle,1))
