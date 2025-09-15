import os
from train.utils import backup_file
from transformers import TrainingArguments
from train.clm_trainner import CLMTrainner, CLMLoRATrainner

if __name__ == "__main__":
    json_path = "./llm-datasets/text/Erotic_Literature_Collection/all_shuffled_10k.json"
    
    output_dir = "./llm-models/output/Qwen3-0.6B-Story"
    model_path = os.path.abspath("./llm-models/Qwen3-0.6B-Base")
    seq_length = 1024
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-5,
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



    backup_file(__file__, output_dir)
    trainer = CLMTrainner(
        model_path=model_path,
        json_path=json_path,
        output_dir=output_dir,
        seq_length=seq_length,
        training_args=training_args,
    )
    trainer.train()
