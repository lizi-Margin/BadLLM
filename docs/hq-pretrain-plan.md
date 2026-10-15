# pt-hq: 小模型 × 高质量小数据集预训练计划

2026-09-30 决策:放弃"更多数据"路线,转向"小而精"数据集。依据 scaling law,114M 模型
在 17B token(65% 为单一来源 web 文本)上训练后 loss 长期平台在 ~2.92(stage-3 全程
8.4 万步几乎无下降),接话无逻辑;且最后阶段(math/code 55%、零 replay)主导输出风格。
结论:先修数据质量,不动模型规模,不进 SFT(PT 尚未记住事实与基本逻辑,SFT 无法激发
不存在的能力)。

## 研究依据

| 结论 | 来源 |
| --- | --- |
| 1M~10M 参数模型在 ~1B 精选 token 上即可产出连贯文本 | TinyStories (Eldan & Li 2023) |
| 教科书级合成数据让 1.3B 模型逼近大 10 倍模型 | phi-1 "Textbooks Are All You Need" |
| 小模型混合大量 LLM 合成教科书文本显著提升 | SmolLM / Cosmopedia v2 (HuggingFaceTB) |
| 数据策展质量 > 原始规模 | DataComp-LM (DCLM) |
| 高质量数据重复 2~4 epoch 接近同量新数据 | "To Repeat or Not To Repeat" (Muennighoff 2023) |

## 数据:pt-hq-2b(约 2B token × 2 epoch = 4B 有效)

池子 = 已完成的 prepared 导出(pt-3b-v4 + pt-3b-v5-reasoning 的 train/validation.jsonl,
合计 ~6B token,构造上互不重复),零新增下载。选择器按 source 内质量分排序取
top-token,配额见 `experiments/qwen3-tiny/data_mix_hq.json`:

| 来源 | 性质 | 目标 | 池内可用 |
| --- | --- | --- | --- |
| finewiki_zh | 百科事实 zh | 340M | 900M |
| fineweb_edu_dedup | 教育 web en | 380M | 750M |
| ultrafineweb_l3_zh_qa | 合成问答 zh | 220M | 600M |
| ultrafineweb_l3_zh_multistyle | 合成多文体 zh | 210M | 300M |
| cosmopedia_v2 | 合成教科书 en | 150M(池上限) | ~150M |
| finewiki_en | 百科事实 en | 140M | 150M |
| ultradata_math_l2/l3 | 数学 | 280M | 1350M |
| ultradata_code_l2/l3 | 代码 | 170M | 1050M |
| ultrafineweb_zh_replay+fresh | web zh(打分 top 切片) | 100M | 750M |

质量分(每 source 内部排序):重复行占比、字符 8-gram 重复度、唯一字符比、行长方差。
构建器 `tools/build_hq_dataset.py` 同时输出 per-domain 验证集(从池子 validation.jsonl
采样,与 train 选择 hash 互斥)。

## 训练(LLaMA-Factory,训练机 RTX 6000D)

```bash
# 1) 构建 HQ 数据集(训练机仓库根目录)
python tools/build_hq_dataset.py \
  --mix experiments/qwen3-tiny/data_mix_hq.json \
  --output data/pretrain/prepared/pt-hq-2b \
  --pool data/pretrain/prepared/pt-3b-v4 \
  --pool data/pretrain/prepared/pt-3b-v5-reasoning

# 2a) 从随机初始化重训(推荐,干净对照):2 epoch ≈ 21h
bash tools/wsl/run-in-env.sh pt llamafactory-cli train \
  experiments/qwen3-tiny/pt-hq-2b-scratch.json

# 2b) 或从当前 checkpoint 退火续训(1 epoch ≈ 10h,lr 5e-5)
bash tools/wsl/run-in-env.sh pt llamafactory-cli train \
  experiments/qwen3-tiny/pt-hq-anneal.json
```

LF 训练中每 2000 步输出 12 个域的 eval_loss(packing/cutoff=1024/bs 32 与旧流程一致;
stage-1 就是 LF 训的,引擎已验证)。

## 评估

1. LF in-run per-domain eval_loss(trainer_state.json / plot_loss)。
2. checkpoint 逐域 PPL 对比(CUDA/MUSA 通用):

```bash
python tools/eval_ppl.py --model <ckpt> --data data/pretrain/prepared/pt-hq-2b \
  --max-tokens 200000 --out eval-<ckpt名>.json
```

对旧 pt-3b-v5-after-v4 和新模型各跑一次,逐域对比;同时用 infer.py 主观对比接话。

## 决策门

- 若 scratch 版 per-domain PPL 全面优于旧模型且接话明显更连贯 → 路线确认;
  可扩 cosmopedia(本地 raw 还有余量)到 300M+ 再加一轮。
- 若仍无逻辑 → 数据再精(去掉 web 切片),或考虑 0.6B 架构(Qwen3-0.6B 已在本地,
  算力约 5 倍)。
- PT 达标前不进 SFT。

## 备注

- MUSA S4000 台机上可跑 tools/eval_ppl.py 与 tools/infer.py(badllm-pt 环境;
  musa_compat shim 已 vendored 到 tools/musa_compat.py,环境内未安装时自动回退,
  启动脚本 tools/start_infer.sh);LF 训练走 mthreads
  官方验证的 v0.9.3 + musify 栈,或仍在 RTX 机训练。
- clone 部署:pip 依赖约束见 experiments/qwen3-tiny/constraints-training.txt,
  LF/verl vendor 树由 tools/wsl/install-environments.sh 按固定 commit 拉取,
  机器相关路径集中在 tools/wsl/badllm-env.sh(或用 BADLLM_PYTHON/BADLLM_UV 覆盖)。
- 构建器/评估器在合成小池上冒烟通过(配额、hash 互斥、跨池去重、垃圾文档排序淘汰、
  LF dataset_info schema)。
