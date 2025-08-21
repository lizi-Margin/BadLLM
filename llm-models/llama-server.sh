#!/bin/zsh
MODEL_PATH="./Qwen2.5-Coder-32B-Instruct-GGUF/qwen2.5-coder-32b-instruct-q4_k_m.gguf"
# MODEL_PATH="./Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"

CURRENT_TIME=$(date +"%Y%m%d %H:%M:%S")

echo "Starting llama-server at $CURRENT_TIME, the model path is $MODEL_PATH" | tee -a ~/.llama-server.log

llama-server \
  --model "${MODEL_PATH}" \
  --ctx-size 8192 \
  --n-gpu-layers 200 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --flash-attn \
  --parallel 1 \
  --threads 20 \
  --host 0.0.0.0 \
  --port 8080 \
  2>&1 | tee -a ~/.llama-server.log

