#!/usr/bin/env bash
set -euo pipefail
source /mnt/d/BadLLM/tools/wsl/badllm-env.sh
mkdir -p "$BADLLM_WORK/vendor" "$BADLLM_REPO/.artifacts/env"
phase=${1:?Usage: install-environments.sh pt|rl}
constraints="$BADLLM_REPO/experiments/qwen3-tiny/constraints-training.txt"
case "$phase" in
  pt)
    environment=badllm-pt
    repo=https://github.com/hiyouga/LLaMA-Factory.git
    commit=95ac3f2373b82662c1bd855c284d3379e6a763d3
    checkout="$BADLLM_WORK/vendor/LLaMA-Factory"
    ;;
  rl)
    environment=badllm-rl
    repo=https://github.com/verl-project/verl.git
    commit=d62da4950573d7a4b7ef2362337952e7ab59e78d
    checkout="$BADLLM_REPO/third_party/verl"
    ;;
  *) exit 2 ;;
esac
# 新机器上 mamba/conda 路径不同时,用 BADLLM_PYTHON / BADLLM_UV 覆盖。
python_path="${BADLLM_PYTHON:-/home/hulc/.local/share/mamba/envs/$environment/bin/python}"
if [ ! -x "$python_path" ]; then
    micromamba create -y -n "$environment" -c conda-forge python=3.11 pip uv
fi
if [ "$phase" = rl ]; then
    # FlashInfer sampling JIT requires nvcc and a CUDA-supported host compiler.
    micromamba install -y -n "$environment" -c conda-forge \
        cuda-version=12.9 cuda-nvcc=12.9 cuda-cudart-dev=12.9 cuda-cccl=12.9 \
        libcurand-dev=10.3.10.19 gxx_linux-64=13
fi
if [ ! -d "$checkout/.git" ]; then
    git init "$checkout"
    git -C "$checkout" remote add origin "$repo"
    git -C "$checkout" fetch --depth=1 origin "$commit"
    git -C "$checkout" checkout --detach FETCH_HEAD
fi
test "$(git -C "$checkout" rev-parse HEAD)" = "$commit"
uv_path="${BADLLM_UV:-/home/hulc/.local/share/mamba/envs/$environment/bin/uv}"
if [ "$phase" = pt ]; then
    "$uv_path" pip install --python "$python_path" -c "$constraints" \
        -e "$checkout" 'trl==0.24.0' tensorboard 'setuptools==79.0.1'
else
    "$uv_path" pip install --python "$python_path" -c "$constraints" \
        -e "$checkout" 'vllm==0.11.0' 'flashinfer-python==0.3.1' \
        'liger-kernel==0.6.2' 'setuptools==79.0.1' 'qwen-vl-utils==0.0.14' mathruler
    "$uv_path" pip install --python "$python_path" --no-deps \
        'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
fi
"$python_path" -m pip check
"$uv_path" pip freeze --python "$python_path" > "$BADLLM_REPO/.artifacts/env/$environment.freeze.txt"
printf '%s environment packages installed. GPU/runtime validation remains.\n' "$environment"
