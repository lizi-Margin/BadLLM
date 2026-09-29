"""Small real GPU checks for the two micromamba environments (not a training run)."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / ".artifacts" / "env"
MODEL = Path("/home/hulc/badllm/smoke/qwen3-114m-init")


def save(stage, result):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"smoke-{stage}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def gpu_check():
    import torch
    assert torch.cuda.is_available()
    assert torch.cuda.is_bf16_supported()
    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    loss = (a @ a.T).float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(a.grad).all()
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "bf16_matmul_backward": True, "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI}


def pt():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Config
    from huggingface_hub import HfApi
    import llamafactory
    result = gpu_check()
    result["llamafactory"] = importlib.metadata.version("llamafactory")
    result["transformers"] = importlib.metadata.version("transformers")
    result["hf_authenticated"] = bool(HfApi().whoami().get("name"))
    spec = json.loads((REPO / "experiments/qwen3-tiny/model_spec.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(spec["tokenizer"]["repo_id"], revision=spec["tokenizer"]["revision"])
    config = Qwen3Config(**spec["config_overrides"], bos_token_id=151643,
                         eos_token_id=151643, pad_token_id=151643)
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    count = sum(p.numel() for p in model.parameters())
    assert count == spec["expected_parameter_count"], (count, spec["expected_parameter_count"])
    assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    MODEL.mkdir(parents=True, exist_ok=True)
    model.generation_config.eos_token_id = [151643, 151645]
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "What is 1 + 1?"}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert "<|im_start|>user" in prompt and "<|im_start|>assistant" in prompt
    model = model.to("cuda")
    inputs = tokenizer("A small language model learns from clear examples. 1 + 1 = 2.", return_tensors="pt").to("cuda")
    loss = model(**inputs, labels=inputs.input_ids).loss
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    result.update(parameters=count, qwen3_bf16_forward_backward=True, loss=loss.item(), model_path=str(MODEL))
    save("pt", result)


def rl():
    import torch
    import flash_attn
    import verl
    import verl.trainer.fsdp_sft_trainer
    import verl.trainer.main_ppo
    import ray
    result = gpu_check()
    result.update(verl=importlib.metadata.version("verl"), verl_source=verl.__file__,
                  vllm=importlib.metadata.version("vllm"), ray=ray.__version__,
                  flash_attn=importlib.metadata.version("flash-attn"))
    q = torch.randn(1, 32, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    out = flash_attn.flash_attn_func(q, k, v, causal=True)
    out.float().square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all() and torch.isfinite(q.grad).all()
    result["flash_attn_gqa_forward_backward"] = True
    ray.init(num_cpus=2, num_gpus=1, include_dashboard=False,
             _node_ip_address="127.0.0.1", object_store_memory=128 * 1024 * 1024)
    try:
        @ray.remote(num_gpus=1)
        def remote_gpu():
            import torch
            x = torch.ones(16, device="cuda", dtype=torch.bfloat16)
            return {"device": torch.cuda.get_device_name(), "sum": x.sum().item()}
        remote_result = ray.get(remote_gpu.remote(), timeout=90)
        assert remote_result["sum"] == 16
        result["ray_gpu_worker"] = remote_result
    finally:
        ray.shutdown()
    save("rl", result)


def vllm_check():
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(MODEL), dtype="bfloat16", tensor_parallel_size=1,
              max_model_len=256, max_num_seqs=2, max_num_batched_tokens=512,
              gpu_memory_utilization=0.15, enforce_eager=True, enable_prefix_caching=False)
    outputs = llm.generate(["Hello", "1 + 1 ="], SamplingParams(max_tokens=8, temperature=0.8), use_tqdm=False)
    assert len(outputs) == 2 and all(o.outputs[0].token_ids for o in outputs)
    save("vllm", {"model": str(MODEL), "requests": len(outputs),
                   "attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND", "auto"),
                   "generated_tokens": [len(o.outputs[0].token_ids) for o in outputs],
                   "note": "Random initialization: validates execution only, not model quality."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["pt", "rl", "vllm"])
    stage = parser.parse_args().stage
    {"pt": pt, "rl": rl, "vllm": vllm_check}[stage]()
