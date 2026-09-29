from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

prompts = ["The capital of France is", "中国的首都是", "2 + 2 =", "Once upon a time"]
for name, path in [("init", "/home/hulc/badllm/models/BadLLM-Qwen3-114M-init"),
                   ("pilot", "/home/hulc/badllm/runs/qwen3-tiny/pt-pilot-10m")]:
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    print("###", name)
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=24, do_sample=False)
        print(prompt, "=>", tokenizer.decode(output[0], skip_special_tokens=True))
    del model
    torch.cuda.empty_cache()
