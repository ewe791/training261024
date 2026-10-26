# 从这里开始（3 步）

> 包内**不含** `data/*.jsonl` 与 `eval/results_*`——这些是生成物，跑一条命令就能重建。
> 所以刚解压时直接跑 `train/finetune_lora.py --dry-run` 会显示「样本数: 0」，那是正常的。

## 现在还没有 GPU 卡？

```bash
cd /root/autodl-tmp/ft-kb-4.1
bash scripts/nocard_prepare.sh        # 装依赖→下模型→配 .env→造数据→体检→自检，全程不需要 GPU
```

约 40–60 分钟、成本约 1 毛钱（无卡模式 ¥0.1/小时）。
详见 [`docs/无卡模式先备环境.md`](docs/无卡模式先备环境.md)（含“系统盘会重置、venv 必须放数据盘”这个坑）。

## 第 1 步：装依赖

```bash
# 云端（AutoDL 等镜像已带 CUDA 版 torch，不要重装）
bash scripts/cloud_setup.sh --no-card

# 本机（需要自己装 torch，RTX 50 系要用 cu128 版，见手册附录 E）
pip install -r requirements.txt
```

> `cloud_setup.sh` 默认建 `.venv` 并加 `--system-site-packages`，
> 目的是让 venv 能看见镜像自带的 torch。少了这个参数，pip 会另装一个（常见后果：装成 CPU 版）。

## 第 2 步：造数据（10 秒，不需要 GPU）

```bash
python scripts/build_dataset.py     # 期望：780 + 251 条，且「标签/表述自洽：通过」
python eval/split_dataset.py        # 期望：train 607/173，test 173/78，覆盖检查通过
```

## 第 3 步：体检 → 训练 → 评测

```bash
python train/finetune_lora.py --dry-run   # 期望：样本数 780，截断 0 条
python train/finetune_lora.py             # 需要 GPU；先配好 .env（见下）
```

`.env` 至少要这两行（`cp .env.example .env`，或让 `nocard_prepare.sh` 自动生成）：

```
FTKB_BACKEND=local
FTKB_BASE_MODEL_PATH=/root/autodl-tmp/Qwen2.5-7B-Instruct     # 换成你的实际路径
```

训练启动后第一屏应打印 `基座模型: …`，且必须是**本地目录**；
如果显示 `Qwen/Qwen2.5-7B-Instruct`，说明 `.env` 没生效，会联网再下一个 15 GB，立刻停掉。

训练完成后评测：

```bash
# .env 里补：FTKB_ADAPTER_PATH=<项目目录>/outputs/lora-knowledge-agent-4.1
python scripts/run_all.py --backend local --skip-build
```

结果在 `reports/eval_finetune.md`。

---

## 完整说明

| 文档 | 什么时候看 |
| --- | --- |
| [`docs/无卡模式先备环境.md`](docs/无卡模式先备环境.md) | **还没 GPU 卡**。无卡模式开机 → 传代码 → 装依赖 → 下模型 → 造数据 → 自检，以及“系统盘重置”的规避方法 |
| [`docs/上云训练详细操作手册.md`](docs/上云训练详细操作手册.md) | **有卡之后**。从"注册账号"写到"释放实例"，含选卡、选镜像、5 种传数据方式、18 条故障排查（1300 行） |
| [`docs/上云微调实施手册.md`](docs/上云微调实施手册.md) | 已经上过云、只想查命令和参数。附录 E 有「本机显卡不够时怎么用」（含 RTX 50 系必须用 cu128 版 torch） |
| [`README.md`](README.md) | 想了解代码结构、数据格式、指标口径、设计取舍 |

## 只想知道当前环境能不能跑

```bash
python scripts/check_env.py              # 有卡环境
python scripts/check_env.py --no-card    # 无卡模式（GPU 检查降级为提示）
```

会告诉你 Python / GPU / 依赖 / 模型 / 数据 / 磁盘的状态，并为每一项给出下一步动作。
