# Windows 原生语料下载

本批使用 Python 标准库下载原始 Parquet，数据固定到 `experiments/qwen3-tiny/data_mix.json` 指定的 Hugging Face revision。下载根目录为 `D:\BadLLM\data\pretrain`，不经过 WSL，不使用 C 盘 Hugging Face 缓存。

状态更新（2026-09-20）：本批 37 个分片、45,099,691,675 bytes 已全部下载，全部通过官方 SHA-256 校验，`progress.json` 为 `status: complete`。下载进程已结束。

## 已固定的首批清单

| 数据源 | 分片数 | 原始文件大小（十进制 GB） |
| --- | ---: | ---: |
| Ultra-FineWeb 中文 | 13 | 16.46 |
| FineWeb-Edu-Dedup | 6 | 14.04 |
| Cosmopedia-v2 | 6 | 7.06 |
| FineWiki 中文 | 5 | 5.53 |
| FineMath-4+ | 7 | 2.00 |
| 合计 | 37 | 45.10 |

分片按固定 seed 从各来源打散选取，FineWiki 中文取当前 revision 的全部 5 个分片。下载量用于提供清洗和 token 配比的原料余量，不等于最终的 40/35/15/5/5 token 配方，也不代表已经得到 3B 或 5B 个有效训练 token。

## 查看进度

在仓库根目录运行 PowerShell：

```powershell
$p = Get-Content data/pretrain/progress.json -Raw | ConvertFrom-Json
$p | Select-Object updated_at, pid, downloaded_bytes, planned_bytes, verified_files, total_files, status
Get-Content data/pretrain/download.stdout.log -Tail 15
Get-Content data/pretrain/download.stderr.log -Tail 15
```

`progress.json` 每 5 秒更新。`downloaded_bytes` 包含未完成的 `.part`；只有 `verified_files` / `verified_bytes` 才表示文件与官方 SHA-256 一致。进度时间使用 UTC。`status: complete` 仅在全部 37 个文件校验通过后写入。

下载已完成或进程退出后，进度文件是最后一次快照。是否仍在运行可读取 `download.pid` 后用 `Get-Process -Id` 核实，不能仅凭 PID 文件判断。

也可直接扫描当前磁盘状态：

```powershell
python tools/download_pretrain.py status
```

此命令是只读快照，JSON 中的 PID 是查询进程自身；后台进程 PID 以 `progress.json` 和 `download.pid` 为准。

## 断点续传

后台任务中断后，在仓库根目录运行：

```powershell
python tools/download_pretrain.py run --workers 3
```

无需重新生成清单。下载器会继续 `.part`，校验完整文件后才改名为 `.parquet`，并写入 `.verified.json`。已校验且大小/修改时间未变化的文件跳过下载；异常文件不伪装成成功。任务有操作系统文件锁，重复启动不会并发写同一批文件。

远端忽略 Range 时安全重下当前分片；SHA-256 不匹配则保留 `.corrupt-*` 供排查。每个分片最多重试 10 次；失败记录到日志和进度，其他分片仍继续。启动时检查剩余下载体积并预留 100 GB 空间，使用电脑期间仍需注意其他程序占用磁盘。

## 文件布局与来源记录

```text
data/pretrain/
  download_plan.json          # 固定 URL/revision/官方 SHA-256/大小/目标路径
  progress.json               # 当前实际落盘量与校验状态
  download.pid
  download.stdout.log
  download.stderr.log
  metadata/<source>/
    README.md                 # 固定 revision 的原始数据卡
    catalog.json              # 该来源的分片目录与元信息
  raw/<source>/
    *.parquet                 # 校验成功的原始语料
    *.parquet.verified.json
    *.parquet.part            # 尚未完成
```

整个 `data/pretrain/` 已加入 Git 忽略。数据卡与分片内的出处字段保留，后续清洗时继续维护来源 sidecar。

## 下一步

2026-09-20 已在 Windows 本地完成五个来源各一个分片的 SHA-256 校验和 PyArrow 正文读取，每源抽样 32 行均为非空文本。检查结果在 `data/pretrain/sample_inspection.json`；这不代表所有分片都已经下载或所有文本质量合格。后续可重复检查：

```powershell
.\.venv-data\Scripts\python.exe tools/inspect_pretrain.py
```

检查工具环境位于仓库内的 `.venv-data`，依赖版本记录在 `experiments/qwen3-tiny/requirements-data.txt`。

下载后读取各源真实 Parquet，检查正文列和样本；用固定 Qwen tokenizer 统计，去重和清理评测污染，再切训练/验证和做 token 配比。原始 Parquet 下载完成不等于训练集已清洗、打包或模型已经训练。

下载器的本地 HTTP 测试覆盖 Range 续传、服务器忽略 Range、错误 Content-Range、损坏文件校验、已完成文件跳过网络和目标路径约束：

```powershell
python -m unittest discover -s tests -p test_download_pretrain.py -v
```
