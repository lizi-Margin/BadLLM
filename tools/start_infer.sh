#!/usr/bin/env bash
# Linux launcher for tools/infer.py on CUDA or MUSA boxes.
# Usage: ./start_infer.sh [--context "text" --max-new-tokens 128 ...]
#        all args are forwarded to infer.py
set -euo pipefail

# 本机 shell 环境里 OMP_NUM_THREADS=0 会让 libgomp 报
# "Invalid value for environment variable OMP_NUM_THREADS"
if [ -z "${OMP_NUM_THREADS:-}" ] || [ "${OMP_NUM_THREADS}" = "0" ]; then
    export OMP_NUM_THREADS=8
fi

# 未在 conda 环境里时,尝试激活 badllm-pt(已在环境里则跳过)。
if [ -z "${CONDA_PREFIX:-}" ]; then
    for init in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/miniforge3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh"; do
        if [ -f "$init" ]; then
            # shellcheck disable=SC1090
            source "$init"
            conda activate badllm-pt
            break
        fi
    done
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "${script_dir}/infer.py" "$@"
