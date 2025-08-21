import os
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

src_model = "./Qwen3-0.6B-Base"
dst_model = "./Qwen3-0.6B-Init"

# 1. 保留 tokenizer
tokenizer = AutoTokenizer.from_pretrained(src_model)

# 2. 读取配置文件
config = AutoConfig.from_pretrained(src_model)

# 3. 随机初始化模型
model = AutoModelForCausalLM.from_config(config)

# 4. 保存
os.makedirs(dst_model, exist_ok=True)
tokenizer.save_pretrained(dst_model)
model.save_pretrained(dst_model)

print(f"✅ 已经将随机初始化的模型保存到 {dst_model}")
