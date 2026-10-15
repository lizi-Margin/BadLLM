from pathlib import Path
import sys

# Usage: python patch_parser.py [LLaMA-Factory repo root]
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/hulc/badllm/vendor/LLaMA-Factory')
p = root / 'src/llamafactory/hparams/parser.py'
s = p.read_text()
old = '    model_args.device_map = {"": get_current_device()}\n'
new = '''    # Avoid device_map during ordinary single-device training. Transformers
    # records it as hf_device_map when loading checkpoints, which makes Trainer
    # misclassify a single-GPU model as model-parallel and enable DataParallel.
    # Trainer will place the complete model on the current CUDA device.
    if model_args.quantization_bit is not None or model_args.use_unsloth or model_args.use_4bit:
        model_args.device_map = {"": get_current_device()}
    else:
        model_args.device_map = None
'''
if old not in s:
    raise SystemExit('target not found')
p.write_text(s.replace(old, new, 1))
