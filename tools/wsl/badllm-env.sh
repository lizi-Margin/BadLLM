#!/usr/bin/env bash
# Source from WSL. Uses the Windows host's existing proxy and HF token file.
# No credentials are copied into the repository or written into logs.
export BADLLM_REPO=/mnt/d/BadLLM
export BADLLM_WORK=/mnt/d/BadLLM/wsl
export HF_HOME="$BADLLM_WORK/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export UV_CACHE_DIR="$BADLLM_WORK/cache/uv"
export HF_TOKEN_PATH=/mnt/c/Users/hulc/.cache/huggingface/token
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export VERL_USE_UV=0
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=8
# WSL NCCL cuMem allocation produced CUDA error 999 during FSDP broadcast.
# Standard allocation passes the single-GPU FSDP2 training/checkpoint test.
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# WSL NAT: localhost on Linux is not Windows localhost.
BADLLM_WINDOWS_HOST=$(ip -4 route show default | awk '{print $3; exit}')
export http_proxy="http://${BADLLM_WINDOWS_HOST}:7898"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"
export no_proxy=localhost,127.0.0.1,::1
export NO_PROXY="$no_proxy"
unset BADLLM_WINDOWS_HOST

# GitHub authentication uses native gh's credential store, initialized from
# Windows gh via a stdin pipe. No token is stored in this script/repository.

export PYTHONPATH=
export PYTHONPATH="/mnt/d/BadLLM/wsl/vendor/LLaMA-Factory/src:/mnt/d/BadLLM/wsl/vendor/verl:${PYTHONPATH:-}"
