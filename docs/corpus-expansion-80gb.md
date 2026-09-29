# 原始语料扩展

2026-09-20 已将原始 Parquet 下载预算从 45.10GB 扩展到 82.42GB。Windows 原生下载器完成 67/67 个分片的 SHA-256 校验，`data/pretrain/progress.json` 状态为 `complete`，无下载错误。

| 来源 | 分片 | 原始 bytes |
| --- | ---: | ---: |
| Ultra-FineWeb 中文 | 25 | 31.67GB |
| FineWeb-Edu-Dedup | 12 | 28.25GB |
| Cosmopedia-v2 | 11 | 12.95GB |
| FineWiki 中文 | 5 | 5.53GB |
| FineMath-4plus | 14 | 4.02GB |
| **合计** | **67** | **82.42GB** |

原始 bytes 不等于有效训练 token：Parquet 压缩、正文列、空/异常文本、exact dedup、验证集隔离和 tokenizer 差异都会造成缩水。下一阶段目标是加工出至少 50GB 有效文本规模，而不是把 82GB 文件直接送入训练。

计划扩展工具：[extend_download_plan.py](../tools/extend_download_plan.py)。原始清洗工具仍使用固定 Qwen tokenizer 和 provenance 记录；正式 50GB 物化前还应使用批量 token 计数和可恢复中间账本，避免中断后从头扫描。
