# BadLLM 新实验：小型 Qwen3 从零预训练与 verl 后训练

更新：2026-09-20。首批 37 个原始分片、45.10 GB 已全部下载并通过 SHA-256 校验，见 [下载说明](corpus-download.md)。两套 micromamba 环境已配置，114M Qwen3、PT 2 steps、verl SFT 1 step、Ray GPU worker 和 vLLM 生成均通过，见 [环境说明](training-environments.md)。正式语料清洗/token 计数与完整 GRPO 更新仍待完成。

## 目标与技术路线

先用约 114M 参数跑通随机初始化 → LLaMA-Factory 全参数预训练 → verl SFT → verl GRPO → Hugging Face 格式导出 → 固定评测。沿用 Qwen3 的架构实现、词表、tokenizer 和聊天格式；所有权重，包括 embedding，重新随机初始化。

首版目标是基本中英文续写、简单指令、简单可验证算术。114M 不应被当成通用助手能力的保证，RL 也不能补回不足的预训练。先验证管线与可测的能力增长，再决定扩展至约 200–300M；不直接重做 0.6B。

当前仓库的 `train_pretrain.py` 实际调用自定义 Transformers Trainer，含旧 Linux 绝对路径。新实验另建目录，预训练明确接入 LLaMA-Factory，不沿用该入口。旧 `train/load_dataset.py` 默认使用 50% 重叠窗口，新实验改为无重叠 packing。

## 模型

设计规格见 [`model_spec.json`](../experiments/qwen3-tiny/model_spec.json)。这是实验规格，不是可直接传给 Transformers 的完整 `config.json`。

| 项目 | 首版配置 |
| --- | --- |
| 架构 | `Qwen3ForCausalLM`，dense，随机初始化 |
| 层数 / hidden / intermediate | 12 / 512 / 1536 |
| attention heads / KV heads / head dim | 8 / 2 / 64 |
| 词表 | 151,936，复用 `Qwen/Qwen3-0.6B` tokenizer |
| embedding 与 LM head | 权重共享 |
| 参数量 | 113,981,440；已实例化核验 |
| 训练精度 | BF16，首次优先标准 SDPA |
| 上下文 | 首轮 1,024；稳定后 2,048 |
| SFT / RL 聊天格式 | 原始 Qwen3 chat template，显式 `enable_thinking=False` |

约 77.8M 参数来自词嵌入，Transformer 部分约 36.2M。因此保留 Qwen 大词表时，继续减少层数的收益有限；输出投影/softmax 仍可能是吞吐瓶颈，114M 不能被直接换算成相对 0.6B 的加速倍数。

tokenizer 来源是现成 Qwen3，模型权重不从它继承。PT 使用原始正文与文档 EOS；SFT/RL 使用 chat template。预训练文档结束与聊天结束 token 分别核对，不能把 `im_end` 与 `endoftext` 随意互换。设置 2,048 上下文不意味着继承原模型的长上下文能力。

## 预训练数据：首版采用五种来源

默认面向中英文通用模型，约 45% 中文 / 55% 英文。下表比例全部按 **Qwen tokenizer 重新统计后的 token 数**，不是文档数、文件大小或上游标注 token 数。比例属于待验证假设。

| 数据 | token 占比 | 3B 首轮预算 | 用途与选择理由 |
| --- | ---: | ---: | --- |
| [OpenBMB/Ultra-FineWeb](https://huggingface.co/datasets/OpenBMB/Ultra-FineWeb)，`default` / `zh` | 40% | 1.20B | 中文主干；已筛选的网页正文，数据卡标称中文约 120B tokens |
| [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)，`fineweb-edu-dedup` | 35% | 1.05B | 英文教育性网页；数据卡标称约 220B tokens，使用其去重子集 |
| 同仓库 `cosmopedia-v2` | 15% | 0.45B | 合成教材、博客、故事；更清晰的教学性文本，但控制占比，避免单一生成风格 |
| [HuggingFaceFW/finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki)，`zh` | 5% | 0.15B | 中文百科，补结构化知识性文章；过滤极短词条和列表页 |
| [HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath)，`finemath-4plus` | 5% | 0.15B | 数学解释与解题正文，数据卡标称约 9.6B tokens，给后续算术 RL 提供基础 |

这些公开语料的规模远大于本次需要。首轮不需要下载全部语料，也不需要先自建大规模合成数据。优先把筛选、去重、混合和评测做可靠。

具体 repo、revision、config、split、正文列和占比已固定在 [`data_mix.json`](../experiments/qwen3-tiny/data_mix.json)。上游规模采用各自 tokenizer，只是来源量级，不能代替我们的训练 token 账本。

### 已核实的读取差异

- Ultra-FineWeb 中文使用 `content` 列，split 是 `zh`，不是 `train`；已确认中文 Parquet 分片存在，前几个分片每个约 1.27 GB。Hub 样本预览服务返回 404，但随后已下载固定 revision 的真实分片，核验 SHA-256，并读取 32 条非空中文正文。预览 404 不代表仓库不可用。
- FineWeb-Edu-Dedup、Cosmopedia-v2、FineWiki-zh、FineMath-4+ 的当前样本预览均已取回，存在 `text`；随后也均完成一个真实分片的本地校验和 32 条正文抽样。尚不能据此认定全部选定分片下载完成。
- Cosmopedia 只训练生成后的 `text`，不把 `prompt` 与 `seed_data` 也拼入预训练正文。
- FineMath 的高分样本仍可能残留低质网页文本；“已筛选”不等于免抽检。
- `python-edu` 已核实只有 `blob_id/repo_name/path/score` 等索引，代码正文需额外从 Software Heritage 取回。首版暂不纳入必需配方；后续有明确代码目标时，再接正文获取与许可记录，替换 5% 英文网页数据做对照。

### 备选，不直接混入首版

- [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) 的普通话配置实际叫 `cmn_Hani`。可作为中文覆盖不足或 Ultra-FineWeb 无法读取时的备选，但不能当成等质量的自动替换。
- [IndustryCorpus2](https://huggingface.co/datasets/BAAI/IndustryCorpus2) 的 Hub 信息标为 gated，需要访问流程；不让它成为最小管线的前提。
- Ultra-FineWeb-L3 提供改写和问答合成数据，可后续单独比较；首轮不同时增加多个合成来源。
- 暂不堆未经筛选的 Common Crawl、全量 The Pile、长篇学术论文和竞赛级数学；也不沿用旧实验的 NSFW 专项配方。

## 数据处理与落盘要求

1. 先每源抽取至少 1,000 篇做语言、长度、乱码、重复段落、网页模板残留检查。跨多个随机分片抽样，不直接取前 N 行代表总体。
2. 统一换行和 Unicode NFC，保留正文结构。中文不过度依赖按空格统计的英文启发式；数学保留 LaTeX；未来代码单独过滤，保留缩进。
3. 全局做规范化文本哈希去重，正式 3B+ 数据再做近重复聚类。对跨来源重复使用同一规则，记录过滤原因和损失 token 数。
4. 清理与固定评测集的重叠，尤其 GSM8K 等题目；合成算术的 train/validation/test 按题目规范形式分开。不能只凭文件名或不同随机种子认定无泄漏。
5. 在 packing **之前**按文档/近重复簇划分训练和验证；每源验证数据单独冻结，目标合计约 1–5M tokens，独立于训练预算。避免同一文档的不同片段跨集合。
6. 用固定 Qwen tokenizer 统计 token，输出按源 JSONL 正文分片和 provenance sidecar；保留来源、revision、URL/原始 ID、哈希、过滤版本与实际 token 数。
7. 按 token 配额抽取、混合并打散，不把文档采样概率直接当成 token 比例。数据耗尽时明确报错或重新制定配方，禁止静默循环灌满配额。
8. 为 LLaMA-Factory 注册正文 `text` 字段，PT 无对话模板包装；由其 tokenizer/packing 路径插入文档边界并生成无重叠训练块。核对跨文档边界和 EOS，没有 50% 滑窗重复。
9. 先以有限分片物化数据，再训练；正式跑尽量不依赖训练中实时访问 HF。HTTP 流式访问仍可能下载一个完整 row group，不能假定取一条只消耗几 KB。

许可证保留在 manifest。Ultra-FineWeb 仓库声明 Apache-2.0，但数据卡同时要求检查上游来源许可；不能把聚合仓库许可证当作所有原始文本的统一授权。

## 预算与验收

| 阶段 | PT token 预算 | 目的 |
| --- | ---: | --- |
| 管线 smoke | 10M | PT、保存/恢复、SFT、GRPO 更新、导出和生成全部能执行；不要求语言能力合格 |
| pilot | 100M | loss 有效下降；测 tokens/s、峰值显存、输入瓶颈；人工检查中英文续写 |
| 首个 base | 3B，必要时延长至 5B | 分来源验证 loss 和基本语言能力稳定改善，再进入有意义的后训练比较 |
| 扩展 | 10B | 仅在验证改善且预算可接受时运行；必要时比较更宽模型 |

这些数值是阶段目标，不是达到可用性的承诺。缩小模型只是降低成本的一部分，质量语料与足够训练量仍必要。正式训练前以相同序列长度和 batch 配置实测稳定吞吐，时间按 `tokens / tokens_per_second` 估算；例如 5B tokens 在 10k tokens/s 下约 5.8 天，在 50k 下约 1.2 天，均不含预处理、评测和后训练。这是算术示例，不是本卡测速结果。

PT 起步：全参数 BF16、1024 context、每步有效 token 约 32k，学习率先以 `3e-4` 作 pilot 起点、cosine、约 2% warmup、gradient clipping 1.0。batch 从小到大实测，分别检查 `3e-4` 与 `6e-4` 的稳定性，不能沿用微调学习率就认定适合随机初始化。不开多卡 DeepSpeed 作为首轮前提，必要时再测 fused/chunked loss 与 FlashAttention 兼容性。

PT 验收：同 tokenizer、同固定验证集上的各源 loss/perplexity；中文/英文续写样例；异常重复率；token 吞吐；断点恢复后的数据进度。看训练 loss 不足以证明通用能力。

## 后训练：verl SFT → GRPO

SFT 也放在 verl，使用固定版本的官方 SFT trainer/示例适配数据。首轮 20k–50k 高质量短样本，后续可扩展 100k–300k。通用对话优先 SmolTalk 中的短对话/约束/改写等合适子集，另补需独立核验的中文指令源和可程序验证的简单算术样本。不能认为英语 SmolTalk 自然补齐中文助手能力。SFT 最大长度先 1024，回答优先简短；明确禁用 thinking 模板。

GRPO 采用单卡、tensor parallel=1、小 rollout batch、每 prompt 4 个采样、短 response，从有确定答案的加减乘除和简单应用题开始。先保证 SFT 模型在任务上有部分成功率，才有机会产生组内不同 reward；全组都错或都对时不应误判为有效训练。

答案评分解析限定答案字段，以正确性为主要奖励；格式奖励不能压过正确性。记录组内 reward 方差、有效组比例、输出长度与无效答案率。先用受限数字语法解析，禁止直接 `eval` 模型生成文本。

单卡 GRPO 仍包含 actor/reference/rollout 等显存开销；vLLM 显存比例从保守值开始，实际核验自定义 Qwen3、小 KV-head 数、tied embeddings 和权重同步。FSDP/vLLM/torch/verl 版本组合在 smoke 成功后冻结。保留 KL 约束并检测奖励投机。

固定相同题目和采样设置，比较 SFT 与 RL 的 held-out exact-match/pass@1，必要时报告 pass@k。另看通用对话是否退化。GSM8K 作为后续外部评测，不能用训练 reward 上升替代泛化结果。最终导出完整 HF 模型与 tokenizer，在独立推理进程中加载验证，不能只保存 verl 的分布式 checkpoint 就宣称完成。

## 本机环境与实际进度

已检查：Windows GPU 为 RTX 6000D，`nvidia-smi` 总显存 85,651 MiB，初次检查时已有约 8 GiB 占用。WSL2 的 archlinux 能识别 GPU。最初 mirrored 网络配置失败；用户修复为 NAT 后，结合 Windows 现有代理已能完成安装与 HF 认证/tokenizer 下载。

Windows 和 WSL 默认 Python 均为 3.14；Windows 另有 uv 管理的 Python 3.11，但检查时没有 torch/transformers/datasets。按用户现有工作流，训练环境统一使用 **WSL micromamba**，不采用 Windows Python 作为训练环境。

已确认 `/usr/sbin/micromamba` 版本为 2.5.0，root prefix 为 `/home/hulc/.local/share/mamba`。已独立创建 `badllm-pt` 和 `badllm-rl`，Python 3.11.16，分别安装 LLaMA-Factory 与 verl，通过 HF checkpoint 衔接。PyTorch 2.8.0+cu128 已通过实际 BF16 运算、模型前后向及 trainer 测试。verl 按用户要求克隆在 `D:\BadLLM\third_party\verl`，editable 安装允许继续改源码。

D 盘检查时约有 1.51 TB 可用；C 盘约 75 GB；WSL 根盘约 737 GiB 可用。HF 缓存/模型/数据明确定位，避免默认写满 C 盘。训练高频读取优先 WSL ext4，Windows 仓库可以保留在 D 盘。

当前完成：首批语料整批下载/校验、五个来源的正文抽样、模型实例化、环境安装和最小 PT/SFT/推理验证。尚未完成：正式语料清洗/token 计数、完整 GRPO 更新与训练评测整链、100M pilot 和 3B 正式训练。最小测试使用合成样本，不代表已得到可用模型。

下一步顺序：各源质量抽检与 tokenizer 计数 → 10M 数据 smoke → PT/SFT/GRPO/导出整链 → 100M 吞吐与质量 pilot → 3B 正式训练。
