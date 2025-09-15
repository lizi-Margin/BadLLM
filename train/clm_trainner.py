from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from .utils import load_model_with_lora, GpuMemoryCallback, GpuMemoryMonitor
from .load_dataset import load_clm_dataset
from trl import SFTTrainer
from peft import LoraConfig
from uhtk.UTIL.colorful import *

class Trainner():
    def __init__(self, **kwargs):
        self.args = {}
        self.args.update(kwargs)

    def set(self, arg, value):
        self.args[arg] = value
    
    def train():
        raise NotImplementedError


class CLMTrainner(Trainner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not "load_dataset_fn" in self.args:
            self.set("load_dataset_fn", load_clm_dataset)

    def train(self):
        (model_path, json_path, output_dir, seq_length, training_args) = (
            self.args["model_path"], self.args["json_path"], self.args["output_dir"], self.args["seq_length"], self.args["training_args"]
        )
        lora_config = None
        if "lora_config" in self.args:
            lora_config = self.args["lora_config"]
        
        load_dataset_fn = self.args["load_dataset_fn"]
        tokenizer, model = load_model_with_lora(model_path, lora_config=lora_config)
        train_dataset, eval_dataset = load_dataset_fn(json_path, tokenizer, max_length=seq_length)

        if eval_dataset is None:
            print黄("Warning: eval_dataset is None")

        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        if lora_config is not None:
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                # tokenizer=tokenizer,
                processing_class=tokenizer,
                data_collator=collator,
                # max_seq_length=seq_length
                # callbacks=[GpuMemoryCallback(device_index=0, every_n_steps=1)],
                peft_config=None,  # 模型已经包装了 LoRA
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                # tokenizer=tokenizer,
                processing_class=tokenizer,
                data_collator=collator,
                # max_seq_length=seq_length,
                # callbacks=[GpuMemoryCallback(device_index=0, every_n_steps=1)],
            )
        
        mem_monitor = GpuMemoryMonitor(device_index=0, log_file="./gpu_mem.log")
        mem_monitor.start()

        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        mem_monitor.stop()

