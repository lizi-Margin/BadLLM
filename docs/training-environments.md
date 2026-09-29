# WSL micromamba 训练环境

2026-09-20 已安装并验证两套环境。使用现有 WSL2 `archlinux` 和 micromamba；不在 Windows 原生 Python 中安装 GPU 训练框架。

## 已安装版本

| 项目 | `badllm-pt` | `badllm-rl` |
| --- | --- | --- |
| Python | 3.11.16 | 3.11.16 |
| PyTorch | 2.8.0+cu128 | 2.8.0+cu128 |
| Transformers | 4.57.1 | 4.57.1 |
| 主框架 | LLaMA-Factory 0.9.4，源码 editable | verl 0.6.1，源码 editable |
| 推理 | Transformers | vLLM 0.11.0 |
| 其他 | TRL 0.24.0 | Ray 2.49.2、FlashAttention 2.8.3、FlashInfer 0.3.1 |
| CUDA 编译工具 | 不需要本地编译器跑本次 PT | Conda CUDA Toolkit 12.9、GCC/G++ 13 |

PyTorch 自带 CUDA runtime 为 **12.8**；RL 环境补充的 **12.9 编译工具链**用于 FlashInfer JIT，两者不是同一个版本字段。驱动为 Windows 提供的 610.88。已在 RTX 6000D（sm_120）验证实际运算，未替换系统驱动。

采用固定 release 和提交以验证这次 Qwen3 小模型流程；没有追踪框架最新 main。

## 源码与存储位置

- verl：Windows `D:\BadLLM\third_party\verl`，WSL `/mnt/d/BadLLM/third_party/verl`。
- verl 分支 `badllm-local`，起点为 `v0.6.1` / `d62da4950573d7a4b7ef2362337952e7ab59e78d`，origin 为 `https://github.com/verl-project/verl.git`。
- `badllm-rl` 实际 import 路径已核实为 `/mnt/d/BadLLM/third_party/verl/verl/__init__.py`。改这里的 Python 源码会在新启动的进程中生效；运行中的 Ray worker 需要重启才会加载修改。
- verl 是独立 Git checkout，父仓库忽略 `third_party/`；后续修改请在该 checkout 单独查看 diff/提交。当前没有修改 verl 上游源码。
- LLaMA-Factory：`/home/hulc/badllm/vendor/LLaMA-Factory`，固定提交 `95ac3f2373b82662c1bd855c284d3379e6a763d3`。
- 环境：`/home/hulc/.local/share/mamba/envs/badllm-pt` 与 `badllm-rl`。
- HF/uv 缓存：`/home/hulc/badllm/cache`；编译缓存也留在 WSL ext4。原始语料继续位于 `D:\BadLLM\data\pretrain`。
- 环境测试模型与 checkpoint：`/home/hulc/badllm/smoke`。这些是随机初始化和极少量合成样本测试产物，不是已训练好的助手模型。

## 使用

WSL 内运行：

```bash
cd /mnt/d/BadLLM
source tools/wsl/badllm-env.sh
micromamba activate badllm-pt
llamafactory-cli version
```

后训练切换为 `micromamba activate badllm-rl`。非交互运行建议使用包装脚本，它会加载项目代理、缓存和 NCCL 设置：

```bash
bash tools/wsl/run-in-env.sh pt llamafactory-cli version
bash tools/wsl/run-in-env.sh rl python -c 'import verl; print(verl.__file__)'
```

从 Windows PowerShell 调用：

```powershell
wsl -d archlinux -- bash /mnt/d/BadLLM/tools/wsl/run-in-env.sh pt llamafactory-cli version
```

## 网络与认证

WSL NAT 网络已能访问 PyPI、Conda 和 GitHub。Hugging Face 直连曾有 TLS 错误；项目脚本通过 NAT 默认网关连接 Windows 已有的 `7898` 代理。代理端口或 WSL 网络模式变化时相应更新 `badllm-env.sh`，不需要修改训练代码。

HF 使用 `HF_TOKEN_PATH=/mnt/c/Users/hulc/.cache/huggingface/token` 直接读取 Windows 的现有认证，已通过 `HfApi().whoami()` 和 tokenizer 下载验证。模型/数据缓存显式放到 WSL 工作目录，避免落入 Windows 默认 C 盘缓存。

当前 WSL 不能直接执行 Windows `gh.exe`（Exec format error），因此已经通过 Windows → WSL 标准输入管道，将已有 GitHub token 导入 WSL 原生 `gh` 的用户凭据存储，并设置 `~/.config/gh/hosts.yml` 权限为 600。没有在仓库或日志保存 token。WSL `gh api user` 已成功验证。如果 Windows token 被轮换，可在 PowerShell 重新导入：

```powershell
& 'C:\Program Files\GitHub CLI\gh.exe' auth token | wsl -d archlinux -- bash -lc 'source /mnt/d/BadLLM/tools/wsl/badllm-env.sh; gh auth login --hostname github.com --git-protocol https --with-token && chmod 600 ~/.config/gh/hosts.yml'
```

## 本机必须保留的运行设置

`badllm-env.sh` 已设置：

```bash
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=8
export VERL_USE_UV=0
```

本机 WSL 上，默认 NCCL cuMem 分配在 FSDP2 参数广播时出现 CUDA error 999；禁用后通过了 SFT 训练、验证和 checkpoint 保存。该设置是本次单卡环境的实测结果，多卡环境需重新验证。

vLLM 使用 FlashAttention attention backend，采样使用 FlashInfer。FlashInfer 首次需要 JIT 编译；仅安装 PyTorch CUDA wheel 不够，必须有 `nvcc`、CUDA/CCCL/curand 开发头文件及受支持的主机编译器。首次运行和生成可能需要几分钟准备算子，不用这段耗时估算正式训练吞吐。

## 已完成的实际验证

| 检查 | 结果 / 证据 |
| --- | --- |
| 两环境 `pip check` | 均为 `No broken requirements found` |
| GPU BF16 矩阵乘法与反向 | 通过，RTX 6000D / compute capability 12.0 |
| 114M Qwen3 | 参数量 113,981,440；共享 embedding；BF16 模型 loss 和梯度有限 |
| LLaMA-Factory PT | 合成文本上实际训练 2 steps，正常保存模型；loss 约 11.88 → 10.50 |
| FlashAttention GQA | head_dim=64、8 query heads / 2 KV heads 的前向与反向通过 |
| Ray GPU worker | 独立 worker 识别 GPU 并完成 BF16 CUDA 运算 |
| verl FSDP2 SFT | 实际训练 1 step，完成验证与模型/优化器/dataloader checkpoint 保存；val loss 11.875 |
| vLLM | 加载同一 114M Qwen3，两条请求各生成 8 tokens |

这些只是环境和框架执行验证。**完整 GRPO 更新、SFT→RL 权重同步、正式语料训练和模型质量尚未验证。** SFT/推理使用随机初始模型来测试运行能力，不能据此宣称已经得到可用模型。

原始证据位于 `.artifacts/env/`：`smoke-pt.json`、`smoke-rl.json`、`smoke-vllm.json`、`trainer-pt.json`、`trainer-sft.json` 及对应日志。环境精确包列表保存为 `badllm-*.freeze.txt` 和 `badllm-*.conda-explicit.txt`。

## 重建与复测

安装脚本会在环境不存在时创建 micromamba Python 3.11 环境，再安装固定提交和约束文件指定的包：

```bash
bash tools/wsl/install-environments.sh pt
bash tools/wsl/install-environments.sh rl
```

脚本会核验源码 HEAD 与预期提交一致，不自动 reset 你修改的代码。后续在 verl 分支提交代码后，不要通过重新安装脚本回退源码；editable 安装会继续使用当前 checkout。

复测命令：

```bash
bash tools/wsl/run-in-env.sh pt python tools/wsl/smoke_environment.py pt
bash tools/wsl/run-in-env.sh rl python tools/wsl/smoke_environment.py rl
bash tools/wsl/run-in-env.sh rl python tools/wsl/smoke_environment.py vllm
bash tools/wsl/run-in-env.sh pt python tools/wsl/smoke_trainers.py pt
bash tools/wsl/run-in-env.sh rl python tools/wsl/smoke_trainers.py sft
```

`smoke_environment.py pt` 会重新生成专用 smoke 初始化模型；不要把正式训练 checkpoint 放在 `/home/hulc/badllm/smoke/qwen3-114m-init`。
