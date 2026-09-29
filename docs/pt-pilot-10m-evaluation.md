# 10M 预训练模型评估

模型：`BadLLM-Qwen3-114M`，随机初始化后在 `pilot-10m-v2` 上进行 1,250 个优化步。训练耗时 546.8 秒，训练集实际经过约 1.02 个 epoch，最终训练 loss 为 7.1862。

## 独立验证集

验证集为 144,115 个有效预测 token；模型和随机初始化基线使用同一 Qwen tokenizer、同一 1024 token 窗口和同一评估脚本。

| 来源 | 初始化 loss | 训练后 loss | 初始化 PPL | 训练后 PPL |
| --- | ---: | ---: | ---: | ---: |
| FineWeb-Edu | 12.0269 | 6.3032 | 167,191 | 546 |
| Ultra-FineWeb 中文 | 12.0306 | 7.5177 | 167,816 | 1,840 |
| Cosmopedia-v2 | 12.0455 | 6.2343 | 170,330 | 510 |
| FineWiki 中文 | 12.0378 | 6.9867 | 169,021 | 1,082 |
| FineMath | 11.9803 | 5.8005 | 159,576 | 330 |
| **整体** | **12.0296** | **6.9934** | **167,638** | **1,089** |

交叉熵下降约 41.9%，PPL 下降约 99.35%。这证明模型已经学到训练语料的统计结构；中文验证 loss 仍高于英文和数学，符合中文数据占比和 tokenizer/模型规模的当前限制。

## 固定提示生成

随机初始化模型会快速重复乱码或异常 token。训练后模型能生成英文和中文的连续文本，但仍有明显重复，数学提示也没有可靠计算能力：

```text
The capital of France is a few time of the world of the world of the world.
中国的首都是在《《《《
2 + 2 = 10000000000000000000000
Once upon a time of the world of the world, and the world of the world
```

这不是可用助手的能力证明。10M token 对 114M 模型只够验证预训练路径和 loss 学习，下一步需要扩大到至少 100M pilot，并加入固定的中文、英文、数学 held-out 任务和去污染检查后再评价泛化。

评估原始结果：`experiments/qwen3-tiny/eval-init.json`、`experiments/qwen3-tiny/eval-pilot-10m.json`。
