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

    use_lora = lora_config is not None
    if use_lora:
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        # model = model.half()
    print绿(f"load model: {model_path}, use_lora={use_lora}")
    return tokenizer, model

def dataset_dir(json_path):
    return [os.path.join(json_path, x) for x in os.listdir(json_path)]


import torch
from transformers import TrainerCallback
class GpuMemoryCallback(TrainerCallback):
    def __init__(self, device_index=0, every_n_steps=100):
        self.device_index = device_index
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps == 0:
            allocated = torch.cuda.memory_allocated(self.device_index) / 1024**2
            reserved = torch.cuda.memory_reserved(self.device_index) / 1024**2
            print(f"\r[step {state.global_step}] allocated={allocated:.2f} MB | reserved={reserved:.2f} MB")

import threading
import time
from datetime import datetime
class GpuMemoryMonitor(threading.Thread):
    def __init__(self, device_index=0, interval=1.2, log_file="./gpu_mem.log"):
        super().__init__()
        self.device_index = device_index
        self.interval = interval
        self.log_file = log_file
        self._stop_event = threading.Event()

    def run(self):
        with open(self.log_file, "a") as f:
            while not self._stop_event.is_set():
                max_allocated = 0
                corresponding_reserved = 0
                for _ in range(10):
                    allocated = torch.cuda.memory_allocated(self.device_index) / 1024**2
                    reserved = torch.cuda.memory_reserved(self.device_index) / 1024**2
                    if allocated > max_allocated:
                        max_allocated = allocated
                        corresponding_reserved = reserved
                    time.sleep(self.interval / 10)  # 将 interval 平均到 10 次采样

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                log_line = f"[{timestamp}] allocated={max_allocated:.2f} MB | reserved={corresponding_reserved:.2f} MB"
                f.write(log_line + "\n")
                f.flush()

    def stop(self):
        self._stop_event.set()
