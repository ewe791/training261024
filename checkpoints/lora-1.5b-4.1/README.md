# LoRA 适配器：kb-agent-4.1（Qwen2.5-1.5B-Instruct）

知识库 Agent 的工具调用 + 回答质量微调产物。**仅用于流程验证，不建议直接交付。**

## ⚠️ 先看这条

本适配器是在 **2026-10-26 数据修订之前** 的数据上训练的（756 条）。
那版数据里 **T3 有 60 条样本缺助手回答**（`messages[-1]` 是用户发言，等于教模型生成用户的话），
已在 `scripts/build_dataset.py` 修复。**用修复后的数据（780 条）重训才能得到有效结果。**

因此本文件的定位是「证明链路跑通」，不是「可用模型」。

## 训练信息

| 项 | 值 |
| --- | --- |
| 基座 | `Qwen/Qwen2.5-1.5B-Instruct` |
| 方法 | LoRA（`peft` 0.21.2），`task_type=CAUSAL_LM` |
| LoRA | r=8，alpha=16，dropout=0.05 |
| 目标模块 | q/k/v/o/gate/up/down_proj |
| 可训练参数 | 9,232,384（占 0.59%） |
| 数据 | 756 条（A 层工具调用 585 + B 层回答质量 171） |
| 超参 | batch 1 × grad_accum 8，lr 1.5e-4 cosine，bf16，梯度检查点 |
| 训练进度 | 200 / 285 步（epoch 2.11），15 s/步，RTX 5060 Laptop 8G 约 54 分钟 |
| 最终 train loss | **0.0077**（第 200 步；第 180 步已降到 0.026，属明显过拟合） |

`trainer_state.json` 里有完整的 loss 曲线（20 个记录点）。

## 文件

| 文件 | 说明 |
| --- | --- |
| `adapter_model.safetensors` | LoRA 权重，36 MB |
| `adapter_config.json` | 适配器配置，`base_model_name_or_path` 已改为 HF ID，可直接挂载 |
| `trainer_state.json` | 训练状态与 loss 曲线（不含权重） |

**没有包含**：`optimizer.pt`（74 MB，只用于续训）、`rng_state.pth`、`scheduler.pt`、
tokenizer 文件（直接用基座自带的）。

## 加载方式

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, "./checkpoints/lora-1.5b-4.1")
```

## 评测结果（不代表当前数据集）

用该适配器在本机跑完整评测，**17 项指标只有 2 项达标**（B1 术语准确率、B1F 格式遵循率），
回归护栏 R2（越权拒答）、R3（JSON 合规）FAIL。

| 指标 | 基座 | 本适配器 |
| --- | --- | --- |
| A 层 JSON 合法率 | 22.6% | 62.9% |
| A1 工具选择 | 44.65% | 32.08% |
| A3 缺参澄清 | 100.00% | 25.00% |
| A5 写操作确认 | 20.00% | 80.00% |
| A6 越权拒答 | 87.50% | 43.75% |
| B1F 格式遵循 | 68.66% | 91.04% |
| B4 引用一致性 | 50.50% | 90.55% |

完整报告见仓库 `reports/eval_finetune.md` 与 `reports/本机1.5B实测记录.md`。
