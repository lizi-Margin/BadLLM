from pathlib import Path
import re
p=Path('/home/hulc/badllm/vendor/LLaMA-Factory/src/llamafactory/hparams/parser.py')
s=p.read_text()
block='''    # Avoid device_map during ordinary single-device training. Transformers
    # records it as hf_device_map when loading checkpoints, which makes Trainer
    # misclassify a single-GPU model as model-parallel and enable DataParallel.
    # Trainer will place the complete model on the current CUDA device.
    if model_args.quantization_bit is not None or model_args.use_unsloth:
        model_args.device_map = {"": get_current_device()}
    else:
        model_args.device_map = None
'''
s=re.sub(r'    # Avoid device_map during ordinary single-device training\..*?    model_args\.device_map = None\n    model_args\.model_max_length', block+'    model_args.model_max_length', s, count=1, flags=re.S)
p.write_text(s)
