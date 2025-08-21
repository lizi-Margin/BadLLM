from datasets import load_dataset

def load_clm_dataset(json_path, tokenizer):
    dataset = load_dataset("json", data_files={"train": json_path})
    split_dataset = dataset["train"].train_test_split(test_size=0.03, seed=1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    def tokenize_fn(examples):
        # return tokenizer(examples["text"], truncation=True)
        return tokenizer(examples["text"], truncation=True, max_length=2200)
        # return tokenizer(examples["text"], truncation=True, max_length=1400)
        # return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    train_dataset = train_dataset.map(tokenize_fn, batched=True, batch_size=200, num_proc=25)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True, batch_size=200, num_proc=25)
    train_dataset = train_dataset.remove_columns(["text"])
    eval_dataset = eval_dataset.remove_columns(["text"])
    # print mean, max, min token num
    trian_dataset_ids_len = [len(ids) for ids in train_dataset['input_ids']]
    eval_dataset_ids_len = [len(ids) for ids in eval_dataset['input_ids']]

    def mean(numbers):
        return int(sum(numbers) / len(numbers))
    
    print(f"train_dataset mean token num: {mean(trian_dataset_ids_len)}")
    print(f"train_dataset max token num: {max(trian_dataset_ids_len)}")
    print(f"train_dataset min token num: {min(trian_dataset_ids_len)}")
    print(f"eval_dataset mean token num: {mean(eval_dataset_ids_len)}")
    print(f"eval_dataset max token num: {max(eval_dataset_ids_len)}")
    print(f"eval_dataset min token num: {min(eval_dataset_ids_len)}")

    return train_dataset, eval_dataset
