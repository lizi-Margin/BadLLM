import torch, os
from utils import backup_file
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from utils import load_model_with_lora
from load_dataset import load_clm_dataset


if __name__ == "__main__":
    json_path = "./llm-datasets/Erotic_Literature_Collection/all_shuffled_10k.json"
    output_dir = "./llm-models/output/Qwen3-0.6B-Story"
    model_path = os.path.abspath("./llm-models/Qwen3-0.6B-Base")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=True,

        eval_strategy="steps",
        eval_steps=500,
        metric_for_best_model="loss",
        greater_is_better=False,
        load_best_model_at_end=False,

        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,

        logging_strategy="steps",
        logging_steps=2,
        report_to="tensorboard",
    )



    backup_file(__file__, output_dir)
    tokenizer, model = load_model_with_lora(model_path)
    train_dataset, eval_dataset = load_clm_dataset(json_path, tokenizer)

    # 4. DataCollator 自动做 causal LM 的 label shift
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # tokenizer=tokenizer,
        processing_class=tokenizer,
        data_collator=collator
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
