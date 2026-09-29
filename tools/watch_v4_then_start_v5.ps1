param(
    [int]$PollSeconds = 60,
    [switch]$Once
)

$ErrorActionPreference = 'Stop'
$repoWin = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$repoWsl = '/mnt/d/BadLLM'
$v4Run = Join-Path $repoWin 'wsl\runs\qwen3-tiny\pt-3b-v4-from-pt50gb-224465'
$v4State = Join-Path $v4Run 'trainer_state.json'
$v4Model = Join-Path $v4Run 'model.safetensors'
$v5Config = Join-Path $repoWin 'experiments\qwen3-tiny\pt-3b-v5-after-v4.json'
$v5Run = Join-Path $repoWin 'wsl\runs\qwen3-tiny\pt-3b-v5-after-v4'
$targetSteps = 91553

if (-not (Test-Path -LiteralPath $v5Config)) { throw "Missing v5 config: $v5Config" }
if (Test-Path -LiteralPath (Join-Path $v5Run 'trainer_state.json')) {
    $existing = Get-Content -LiteralPath (Join-Path $v5Run 'trainer_state.json') -Raw | ConvertFrom-Json
    if ([double]$existing.global_step -gt 0) {
        throw "v5 already has training state at $($existing.global_step) steps: $v5Run"
    }
}

Write-Host "Watching v4 until global_step >= ${targetSteps}: $v4Run"
while ($true) {
    $ready = $false
    $step = 0
    if ((Test-Path -LiteralPath $v4State) -and (Test-Path -LiteralPath $v4Model)) {
        try {
            $state = Get-Content -LiteralPath $v4State -Raw | ConvertFrom-Json
            $step = [int]$state.global_step
            $ready = ($step -ge $targetSteps)
        } catch {
            Write-Host "v4 state is still being written; waiting..."
        }
    }
    if (-not $ready) {
        $latest = Get-ChildItem -LiteralPath $v4Run -Directory -Filter 'checkpoint-*' -ErrorAction SilentlyContinue |
            ForEach-Object {
                $s = Join-Path $_.FullName 'trainer_state.json'
                if (Test-Path -LiteralPath $s) {
                    try {
                        $j = Get-Content -LiteralPath $s -Raw | ConvertFrom-Json
                        [pscustomobject]@{ Step = [int]$j.global_step; Name = $_.Name }
                    } catch { }
                }
            } | Sort-Object Step -Descending | Select-Object -First 1
        if ($latest) { $step = $latest.Step }
    }
    if ($ready) { break }
    Write-Host "[$(Get-Date -Format s)] v4 step $step / $targetSteps; next check in ${PollSeconds}s"
    if ($Once) { exit 0 }
    Start-Sleep -Seconds $PollSeconds
}

Write-Host "v4 is complete at step $step. Starting v5 from the final v4 model."
$command = "source $repoWsl/tools/wsl/badllm-env.sh; export CUDA_VISIBLE_DEVICES=0; cd $repoWsl; micromamba run -n badllm-pt python $repoWsl/tools/train_pretrain_v3.py --config $repoWsl/experiments/qwen3-tiny/pt-3b-v5-after-v4.json"
wsl -d archlinux -- bash -lc $command
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
