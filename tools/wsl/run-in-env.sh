#!/usr/bin/env bash
set -euo pipefail
source /mnt/d/BadLLM/tools/wsl/badllm-env.sh
case "${1:-}" in
  pt) badllm_environment=badllm-pt ;;
  rl) badllm_environment=badllm-rl ;;
  *) printf 'Usage: run-in-env.sh pt|rl command [args...]\n' >&2; exit 2 ;;
esac
shift
exec micromamba run -n "$badllm_environment" "$@"
