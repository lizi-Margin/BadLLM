import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# -------------------------
# 模型交互封装
# -------------------------
class Model:
    def __init__(self, model_name: str, device: str = None):
        print(f"Loading model {model_name} ... (this may take a while)")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 如果模型在 HuggingFace 上需要 trust_remote_code，可在 from_pretrained 中传入 trust_remote_code=True
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to(self.device)

    def generate_once(self, context, **gen_kwargs) -> str:
        if not hasattr(self.tokenizer, "apply_chat_template"): raise AttributeError("apply_chat_template not found")
        inputs = self.tokenizer.apply_chat_template(
            context,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True. (thinking does not work)
        )
        prompt_len = inputs.shape[1]
        # streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=False)

        with torch.no_grad():
            out = self.model.generate(
                inputs.to(self.device),
                **gen_kwargs
            )
        # 只解码新生成的部分
        generated = self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        return generated