from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from utils import load_model_with_lora
from load_dataset import load_clm_dataset
from trl import SFTTrainer
from peft import LoraConfig


class CLMTrainner():
    def __init__(self, **kwargs):
        self.args = {}
        self.args.update(kwargs)

    def set(self, arg, value):
        self.args[arg] = value

    def train(self):
        (model_path, json_path, output_dir, seq_length, training_args) = (
            self.args["model_path"], self.args["json_path"], self.args["output_dir"], self.args["seq_length"], self.args["training_args"]
        )
        
        tokenizer, model = load_model_with_lora(model_path)
        train_dataset, eval_dataset = load_clm_dataset(json_path, tokenizer, max_length=seq_length)

        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            # tokenizer=tokenizer,
            processing_class=tokenizer,
            data_collator=collator,
            # max_seq_length=seq_length
        )

        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

class CLMLoRATrainner(CLMTrainner):
    def train(self):
        (model_path, json_path, output_dir, seq_length, training_args) = (
            self.args["model_path"], self.args["json_path"], self.args["output_dir"], self.args["seq_length"], self.args["training_args"]
        )
        lora_config = self.args["lora_config"]

        tokenizer, model = load_model_with_lora(model_path, lora_config)
        train_dataset, eval_dataset = load_clm_dataset(json_path, tokenizer, max_length=seq_length)


        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=None,  # 模型已经包装了 LoRA
            # tokenizer=tokenizer,
            processing_class=tokenizer,
            args=training_args,
            data_collator=collator,
            # max_seq_length=seq_length
        )

        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

