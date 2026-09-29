# 10M 预训练语料 v2

生成日期：2026-09-20。位置：`data/pretrain/prepared/pilot-10m-v2/`。
这是 pilot 数据集，尚未做近重复聚类和 benchmark 污染筛查，不作为正式能力评测的无污染保证。

| 来源 | 训练 tokens | 文档 |
| --- | ---: | ---: |
| Ultra-FineWeb 中文 | 4,012,796 | 4,260 |
| FineWeb-Edu-Dedup | 3,500,598 | 3,689 |
| Cosmopedia-v2 | 1,500,109 | 2,159 |
| FineWiki 中文 | 500,199 | 933 |
| FineMath-4plus | 500,379 | 328 |
| 合计 | 10,014,081 | 11,369 |

独立验证集：144,327 tokens、130 篇文档。计数包含每篇一个 EOS，尚未扣除 packing 尾部损失。
每源保留最后一篇完整文档，允许一篇文档大小的配额超额；验证集目标为 100K，超额主要来自一篇中文长文。
验证集样本仍较少，特别是数学只有 4 篇，不适合据此判断能力。

## 本次修正

- v1 为了填满最后几十个 token 持续扫描，之前将此归因于 tokenizer/磁盘瓶颈不准确。v2 到达配额后立即停止；v1 也已修正。
- 批次内重复与跨源重复统一交给 SQLite 唯一键处理。
- 按 seed 随机分片顺序、row group 和行，轮流读取分片，避免只取第一个分片前 N 行。小配额仍可能只覆盖少量分片；本版 FineMath 仅覆盖一个分片。
- 复用基础文本清洗，新增重复行占比和正文特殊 token 过滤。
- 训练/验证由规范化正文哈希稳定分流，沿用 v1 的规则。跨运行相同文档不会改变 split；验证文档名单尚非跨版本完全冻结。
- 每篇保存文件、row group、行号、repo/revision、原始 SHA256，以及源中存在的 URL/ID/title。保留源许可说明。
- 输出前跨来源打乱，生成文件 SHA256、实际 token 账本和 LLaMA-Factory dataset_info。

## 重现与校验

输出目录必须不存在，防止覆盖已有数据。使用新的输出目录重现：

```powershell
wsl -d archlinux -- bash /mnt/d/BadLLM/tools/wsl/run-in-env.sh pt python /mnt/d/BadLLM/tools/prepare_pretrain_v2.py --output /mnt/d/BadLLM/data/pretrain/prepared/pilot-10m-repro --tokens 10000000
wsl -d archlinux -- bash /mnt/d/BadLLM/tools/wsl/run-in-env.sh pt python /mnt/d/BadLLM/tools/audit_pretrain.py /mnt/d/BadLLM/data/pretrain/prepared/pilot-10m-repro
```

审计逐文档重新 tokenize，核验哈希、唯一性、split、各来源计数和输出 SHA256，成功后写入 `audit.json`。
v1 中断遗留的 `pilot-10m`、`smoke-10m` 等目录不应用于训练；使用有完成 manifest 和通过 audit 的版本。
本次未启动新训练。旧的 1M smoke 指标不能当作此数据集的训练结果。

下一阶段扩大至 100M 时应增加独立验证预算、冻结具体验证文档，并补 benchmark 去污染和近重复检查；训练配置还需显式核对文档 EOS（旧 smoke 自动选择了 im_end）。
