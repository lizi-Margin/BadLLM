from pathlib import Path
import sys

# Usage: python patch_loader.py [LLaMA-Factory repo root]
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/mnt/d/BadLLM/wsl/vendor/LLaMA-Factory')
p = root / 'src/llamafactory/data/loader.py'
s = p.read_text()
s = s.replace('streaming=data_args.streaming and dataset_attr.load_from != "file",', 'streaming=data_args.streaming,')
s = s.replace('''        if data_args.streaming and dataset_attr.load_from == "file":
            dataset = dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)
''', '')
p.write_text(s)
