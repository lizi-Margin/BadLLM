import os
import torch
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig
from trl import SFTTrainer
from utils import backup_file, load_model_with_lora
from load_dataset import load_clm_dataset



if __name__ == "__main__":
    json_path = "./llm-datasets/Erotic_Literature_Collection/all_shuffled_10k.json"
    output_dir = "./llm-models/output/Qwen3-0.6B-Story"
    model_path = os.path.abspath("./llm-models/Qwen3-0.6B-Base")
    seq_length = 2200
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        # fp16=True,
        bf16=True,

        eval_strategy="steps",
        eval_steps=200,
        metric_for_best_model="loss",
        greater_is_better=False,
        load_best_model_at_end=False,

        save_strategy="steps",
        save_steps=200,
        save_total_limit=5,

        logging_strategy="steps",
        logging_steps=2,
        report_to="tensorboard",
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        # target_modules=["q_proj", "v_proj"],
        # target_modules=["q_proj", "k_proj", "v_proj"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )



    backup_file(__file__, output_dir)
    tokenizer, model = load_model_with_lora(model_path, lora_config)
    train_dataset, eval_dataset = load_clm_dataset(json_path, tokenizer, max_length=seq_length)


    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=None,  # 模型已经包装了 LoRA
        tokenizer=tokenizer,
        # processing_class=tokenizer,
        args=training_args,
        data_collator=collator,
        max_seq_length=seq_length
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
