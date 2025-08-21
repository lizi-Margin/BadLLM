import os
from utils import backup_file, dataset_dir
from transformers import TrainingArguments
from clm_trainner import CLMTrainner
from load_dataset import load_big_pretraining_trainset

if __name__ == "__main__":
    # json_path = dataset_dir("./llm-datasets/text_pretrain")
    json_path = dataset_dir("/root/autodl-tmp/llm-datasets/text_pretrain")
    print(json_path)
    output_dir = "./llm-models/output/BadLLM3-0.6B-Story"
    model_path = os.path.abspath("./llm-models/Qwen3-0.6B-Init")
    seq_length = 1024
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=14,
        gradient_accumulation_steps=6,
        # num_train_epochs=12,
        max_steps=200_000, 
        learning_rate=1e-4,
        lr_scheduler_type="cosine",    # cosine decay
        warmup_ratio=0.03,     
        # fp16=True,
        bf16=True,

        # eval_strategy=None,
    
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=5,

        logging_strategy="steps",
        logging_steps=100,
        report_to="tensorboard",
    )



    backup_file(__file__, output_dir)
    trainer = CLMTrainner(
        model_path=model_path,
        json_path=json_path,
        output_dir=output_dir,
        seq_length=seq_length,
        training_args=training_args,
        load_dataset_fn=load_big_pretraining_trainset,
    )
    trainer.train()
