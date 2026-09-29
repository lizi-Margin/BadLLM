param(
  [string]$Config = "/mnt/d/BadLLM/experiments/qwen3-tiny/pt-full-50gb-streaming-resume.json"
)

$envScript = "/mnt/d/BadLLM/tools/wsl/badllm-env.sh"
wsl -d archlinux -- bash -lc "source $envScript; export CUDA_VISIBLE_DEVICES=0; test -f /mnt/d/BadLLM/data/pretrain/prepared/full-50gb-v3/manifest.json || { echo 'Preprocessing is not complete: manifest.json is missing.'; exit 2; }; micromamba run -n badllm-pt llamafactory-cli train $Config"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }




