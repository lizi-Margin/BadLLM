$ErrorActionPreference = 'Stop'
$repo = '/mnt/d/BadLLM'
$config = "$repo/experiments/qwen3-tiny/pt-3b-v4-from-pt50gb.json"
$script = "$repo/tools/train_pretrain_v3.py"
$command = "source $repo/tools/wsl/badllm-env.sh; export CUDA_VISIBLE_DEVICES=0; cd $repo; nice -n 10 micromamba run -n badllm-pt python $script --config $config"
wsl -d archlinux -- bash -lc $command
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
