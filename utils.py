import os, torch
from uhtk.UTIL.colorful import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def backup_file(filename, to_dir):
    os.makedirs(to_dir, exist_ok=True)
    import shutil
    to = os.path.join(to_dir, os.path.basename(filename) + ".backup")
    if os.path.exists(to):
        print黄(f"Warning: {to} already exists, will be removed")
        os.remove(to)
    
    shutil.copy(filename, to)
    print(f"Backup {filename} to {to}")


def load_model_with_lora(model_path, lora_config: LoraConfig = None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        print("pad_token is None")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        # torch_dtype=torch.float16
    )
    print(model)

    if lora_config:
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        # model = model.half()
    print绿(f"load model: {model_path}, lora_config: {lora_config}")
    return tokenizer, model

