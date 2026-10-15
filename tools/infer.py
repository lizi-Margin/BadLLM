#!/usr/bin/env python3
"""Interactive text continuation for a Qwen3 pre-training checkpoint."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 摩尔线程 MUSA 平台支持:torch_musa 注册 torch.musa 命名空间,
# musa_compat(优先 badllm-pt 的 site-packages,缺失时回退到本目录的 vendored 副本)
# 补齐 transformers 需要的缺失 API,并绕开 MUSA torch.cat 拒绝 1-D 空 tensor 混拼的问题
# (DynamicCache 首次 update 会触发,不开 shim 则 KV cache 生成必崩:
#  RuntimeError: Wrong Cat dim: 2)。在 CUDA 机器上这些 import 都是无操作。
try:
    import torch_musa  # noqa: F401

    HAS_MUSA = torch.musa.is_available()

    if HAS_MUSA:
        try:
            import musa_compat  # noqa: F401
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import musa_compat  # noqa: F401
except ImportError:
    HAS_MUSA = False


def latest_checkpoint(root: Path) -> Path:
    candidates = [p for p in root.glob("checkpoint-*") if p.is_dir()]
    numbered = []

    for path in candidates:
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match:
            numbered.append((int(match.group(1)), path))

    if not numbered:
        raise FileNotFoundError(f"No checkpoint-* directory found under {root}")

    return max(numbered, key=lambda item: item[0])[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context",
        default=None,
        help="Initial text prefix",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (default: latest checkpoint under cwd, "
        "falling back to this script's directory)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="musa / cuda / cpu / auto (default: auto)",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    root = cwd if any(cwd.glob("checkpoint-*")) else script_dir

    checkpoint = (
        Path(args.checkpoint).expanduser()
        if args.checkpoint
        else latest_checkpoint(root)
    )

    if not checkpoint.is_absolute():
        checkpoint = (root / checkpoint).resolve()

    if args.device == "auto":
        if HAS_MUSA:
            device_map = "musa"
        elif torch.cuda.is_available():
            device_map = "cuda"
        else:
            device_map = None
    else:
        device_map = None if args.device == "cpu" else args.device

    print(f"Loading checkpoint: {checkpoint}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        trust_remote_code=True,
    )

    # 注意:不能写死 device_map="cuda"。MUSA 版 torch 没有编译 CUDA,
    # transformers 的 caching_allocator_warmup 会走 torch.cuda.current_device()
    # 直接 AttributeError;device.type 决定用哪个加速模块,所以 MUSA 上必须是 "musa"。
    load_kwargs = {"trust_remote_code": True}
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        **load_kwargs,
    )

    model.eval()
    device = next(model.parameters()).device
    print(f"Device: {device}", flush=True)

    # 保存完整文本上下文，而不是 chat messages。
    context = args.context or ""

    def continue_text(prefix: str) -> str:
        inputs = tokenizer(
            prefix,
            return_tensors="pt",
            add_special_tokens=True,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        input_len = inputs["input_ids"].shape[-1]

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            # config.json 里 use_cache=false 是训练时 gradient checkpointing 的遗留,
            # 推理必须显式打开,否则每步重算整个前缀。
            "use_cache": True,
        }

        if args.temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated_ids = output[0, input_len:]

        continuation = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return continuation

    if context:
        continuation = continue_text(context)

        print("\n续写:\n", end="")
        print(continuation, flush=True)

        context += continuation

    print(
        "\n输入文本让模型继续写；"
        "/reset 清空上下文；"
        "/show 查看当前上下文；"
        "/quit 退出。",
        flush=True,
    )

    while True:
        try:
            text = input("\n输入: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if text.strip() in {"/quit", "/exit"}:
            break

        if text.strip() == "/reset":
            context = ""
            print("上下文已清空。", flush=True)
            continue

        if text.strip() == "/show":
            print("\n----- Context -----")
            print(context)
            print("-------------------")
            continue

        if not text:
            continue

        # 如果已有上下文，就原样往后追加。
        context += text

        continuation = continue_text(context)

        print("\n续写:\n", end="")
        print(continuation, flush=True)

        # 模型生成内容也进入下一轮上下文。
        context += continuation


if __name__ == "__main__":
    main()
