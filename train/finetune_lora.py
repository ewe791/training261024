"""F2 —— LoRA 微调训练脚本。

数据组织：
  data/train.jsonl（A 层工具调用）+ data/quality_train.jsonl（B 层问答，按 quality_ratio 混入）
  —— 每条样本的 messages 已含 system/user/assistant，assistant 内容即统一输出协议 JSON 或问答文本。

两种运行模式：
  python train/finetune_lora.py --dry-run      # 只做数据与 token 统计，不需要 GPU
  python train/finetune_lora.py                # 真实训练（需要 transformers/peft/datasets/torch）

依赖见 requirements.txt（train 段）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import config  # noqa: E402
from common.dataio import read_jsonl  # noqa: E402


def read_yaml_flat(path: Path) -> dict:
    out: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k.strip()] = v.lower() == "true"
        else:
            try:
                out[k.strip()] = float(v) if ("." in v or "e" in v.lower()) else int(v)
            except ValueError:
                out[k.strip()] = v
    return out


def load_dataset_rows(cfg: dict) -> list[dict]:
    tool_rows = read_jsonl(ROOT / cfg["train_file"])
    quality_rows = read_jsonl(ROOT / cfg["quality_file"])
    ratio = float(cfg.get("quality_ratio", 0.25))
    # 质量数据按比例混入，避免被 A 层大样本淹没
    n_q = min(len(quality_rows), int(len(tool_rows) * ratio / max(1 - ratio, 1e-6)))
    rows = tool_rows + quality_rows[:n_q]
    return rows


def to_chat(tokenizer, sample: dict) -> dict:
    messages = sample["messages"]
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    return {"input_ids": full_ids, "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]}


def dry_run(cfg: dict) -> int:
    rows = load_dataset_rows(cfg)
    if not rows:
        # 刚解压的代码包里没有 data/*.jsonl（生成物不进包），这里必须说清楚
        print("== 训练数据体检 ==")
        print("样本数: 0")
        print()
        print("数据还没生成。先跑这两步（10 秒，不需要 GPU）：")
        print("  python scripts/build_dataset.py")
        print("  python eval/split_dataset.py")
        print()
        print(f"检查路径是否正确：train_file={cfg.get('train_file')} "
              f"quality_file={cfg.get('quality_file')}")
        return 2
    cats: dict[str, int] = {}
    agents: dict[str, int] = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
        agents[r["agent"]] = agents.get(r["agent"], 0) + 1
    lens = [sum(len(m["content"]) for m in r["messages"]) for r in rows]
    # 中文约 1.4 字符/token（Qwen BPE）；字符数不能直接和 max_length(token) 比
    est_tokens = [int(x / 1.4) for x in lens]
    max_len = int(cfg.get("max_length", 2048))
    n_trunc = sum(1 for t in est_tokens if t > max_len)
    print("== 训练数据体检 ==")
    print(f"样本数: {len(rows)}")
    print(f"类别: {json.dumps(cats, ensure_ascii=False)}")
    print(f"agent: {json.dumps(agents, ensure_ascii=False)}")
    print(f"字符长度: min={min(lens)} max={max(lens)} avg={sum(lens) // len(lens)}")
    print(f"token 估算(字符/1.4): 平均 {sum(est_tokens) // len(est_tokens)}，"
          f"最大 {max(est_tokens)}")
    print(f"LoRA: rank={cfg.get('lora_rank')} alpha={cfg.get('lora_alpha')} "
          f"lr={cfg.get('learning_rate')} epochs={cfg.get('num_epochs')}")
    print(f"max_length={max_len} → 估算截断 {n_trunc} 条"
          f"（{n_trunc / len(rows):.1%}）；超过 5% 建议调大 max_length 或接受截断")

    # 首条样本回显，便于人工确认协议与 label mask 正确
    first = rows[0]
    print("\n== 样本示例 ==")
    for m in first["messages"]:
        print(f"[{m['role']}] {m['content'][:160]}")
    try:
        from transformers import AutoTokenizer  # noqa: F401
    except Exception:
        print("\n（未安装 transformers，跳过 token 统计；真实训练前请 pip install -r requirements.txt）")
    return 0


def resolve_base_model(arg_base: str | None, cfg: dict) -> str:
    """基座路径优先级：命令行 > FTKB_BASE_MODEL_PATH（.env）> 配置文件。

    云端如果在配置里留 HF 模型 ID，训练脚本会去联网再下一个 15 GB 的权重；
    .env 配一次本地目录，训练与评测就都走本地。
    """
    env_base = os.getenv("FTKB_BASE_MODEL_PATH", "").strip()
    return arg_base or env_base or cfg.get("base_model", "")


def train(cfg: dict, args) -> int:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    base_model = resolve_base_model(args.base_model, cfg)
    if not Path(base_model).exists():
        print(f"[提示] 基座 `{base_model}` 不是本地目录，将按模型 ID 联网拉取。"
              f"云端请先下载并用 FTKB_BASE_MODEL_PATH 指向它。")
    print(f"基座模型: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_dataset_rows(cfg)
    samples = []
    max_len = int(cfg.get("max_length", 2048))
    for r in rows:
        item = to_chat(tokenizer, r)
        item["input_ids"] = item["input_ids"][:max_len]
        item["labels"] = item["labels"][:max_len]
        samples.append(item)
    ds = Dataset.from_list(samples)
    print(f"训练样本 {len(ds)} 条，avg tokens≈{sum(len(x['input_ids']) for x in samples) // len(samples)}")

    use_4bit = bool(cfg.get("load_in_4bit"))
    device_map = "auto" if torch.cuda.is_available() else None
    if use_4bit:
        # QLoRA：4bit 量化加载，显存占用约 1/3（12–16G 卡可用）
        from transformers import BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            ),
            device_map=device_map, trust_remote_code=True,
        )
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, trust_remote_code=True, device_map=device_map,
        )

    lora = LoraConfig(
        r=int(cfg.get("lora_rank", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=[m.strip() for m in str(cfg["target_modules"]).split(",") if m.strip()],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    # LoRA + gradient_checkpointing 的固定搭配：不打开这个开关，
    # 反向传播时输入没有梯度，会报 "element 0 of tensors does not require grad"
    # 或者训完发现 LoRA 权重几乎没变（loss 一直平）。
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    out_dir = ROOT / str(cfg.get("output_dir", "outputs/lora-4.1"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # bf16 需要 Ampere 及以上（A100/3090/4090）；T4/V100 等只支持 fp16
    supports_bf16 = (torch.cuda.is_available()
                     and torch.cuda.get_device_capability(0)[0] >= 8)
    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(cfg.get("batch_size", 2)),
        gradient_accumulation_steps=int(cfg.get("grad_accum", 8)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        num_train_epochs=float(cfg.get("num_epochs", 3)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(cfg.get("lr_scheduler", "cosine")),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_steps=int(cfg.get("save_steps", 200)),
        save_total_limit=2,
        bf16=supports_bf16,
        fp16=torch.cuda.is_available() and not supports_bf16,
        gradient_checkpointing=True,
        report_to=[],
        remove_unused_columns=False,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt"
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    resume = bool(getattr(args, "resume", False))
    trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"\nLoRA 适配器已保存：{out_dir}")
    print("下一步：python train/merge_lora.py（可选合并）→ python eval/test_tool_calling.py")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(config.CONFIG_DIR / "lora_sft.yaml"))
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="从 output_dir 里最新的 checkpoint 续训（竞价实例被回收后用）")
    args = ap.parse_args()

    cfg = read_yaml_flat(Path(args.config))
    if args.dry_run:
        return dry_run(cfg)
    return train(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
