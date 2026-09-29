from pathlib import Path
p=Path('/mnt/d/BadLLM/wsl/vendor/LLaMA-Factory/src/llamafactory/data/loader.py')
s=p.read_text()
s=s.replace('streaming=data_args.streaming and dataset_attr.load_from != "file",','streaming=data_args.streaming,')
s=s.replace('''        if data_args.streaming and dataset_attr.load_from == "file":
            dataset = dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)
''','')
p.write_text(s)
