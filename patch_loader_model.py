from pathlib import Path
p=Path('/mnt/d/BadLLM/wsl/vendor/LLaMA-Factory/src/llamafactory/model/loader.py')
s=p.read_text()
old='''    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
'''
new='''    # A checkpoint loaded with a one-entry device map can retain
    # ``hf_device_map`` even though every parameter is on the single current
    # device. Transformers Trainer treats that metadata as model parallel and
    # silently switches to DataParallel, shrinking the configured batch. Clear
    # the stale metadata for ordinary single-device training.
    if is_trainable and getattr(model, "hf_device_map", None) is not None:
        devices = set(model.hf_device_map.values())
        if len(devices) <= 1 and all(str(d) not in {"cpu", "disk"} for d in devices):
            delattr(model, "hf_device_map")
            model.is_parallelizable = False
            model.model_parallel = False

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
'''
if old not in s: raise SystemExit('target missing')
p.write_text(s.replace(old,new,1))
