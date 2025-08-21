from datasets import load_dataset
from uhtk.UTIL.colorful import *

N_CPU = 32

def load_big_pretraining_trainset(file_list, tokenizer, max_length=2048, stride=None):
    window_size = max_length
    if stride is None:
        stride = window_size // 2  # 50% 重叠

    dataset = load_dataset("json", data_files=file_list, streaming=True)

    def tokenize_and_chunk(example):
        input_ids = tokenizer(example["text"], truncation=False)["input_ids"]
        chunks = []
        start_index = 0
        while start_index < len(input_ids):
            end_index = min(start_index + window_size, len(input_ids))
            chunks.append({"input_ids": input_ids[start_index:end_index]})
            if end_index == len(input_ids):
                break
            start_index += stride
        return chunks

    dataset = dataset["train"].map(tokenize_and_chunk, batched=False, num_proc=N_CPU)
    return dataset, None


def load_clm_dataset(json_path, tokenizer, max_length=None):
    eos_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={"train": json_path})
    split_dataset = dataset["train"].train_test_split(test_size=0.03, seed=1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    if max_length is None:
        print绿("use tokenizer max_length = None") 
        def tokenize_fn(examples):
            examples["text"] = [text + eos_token for text in examples["text"]]
            return tokenizer(examples["text"], truncation=True)
    else:
        print绿(f"use tokenizer max_length = {max_length}") 
        print黄("eos_token may be cut by max_length")
        def tokenize_fn(examples):
            examples["text"] = [text + eos_token for text in examples["text"]]
            return tokenizer(examples["text"], truncation=True, max_length=max_length)
            # return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    train_dataset = train_dataset.map(tokenize_fn, batched=True, batch_size=200, num_proc=N_CPU)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True, batch_size=200, num_proc=N_CPU)
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
