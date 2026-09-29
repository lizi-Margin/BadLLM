param([switch]$Smoke)
$script = '/mnt/d/BadLLM/tools/train_pretrain_v3.py'
$extra = if ($Smoke) { '--smoke' } else { '' }
wsl -d archlinux -- bash -lc "source /mnt/d/BadLLM/tools/wsl/badllm-env.sh; export CUDA_VISIBLE_DEVICES=0; micromamba run -n badllm-pt python $script $extra"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
