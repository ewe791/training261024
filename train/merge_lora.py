"""合并 LoRA 适配器到基座权重（可选，推理更省显存/更快）。

运行：
  python train/merge_lora.py \
      --base-model Qwen/Qwen2.5-7B-Instruct \
      --adapter outputs/lora-knowledge-agent-4.1 \
      --out outputs/merged-4.1
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=None,
                    help="基座目录；不传则读 .env 的 FTKB_BASE_MODEL_PATH")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    base = (args.base_model or os.getenv("FTKB_BASE_MODEL_PATH", "")).strip()
    if not base:
        raise SystemExit(
            "缺少基座路径：传 --base-model，或在 .env 里设置 FTKB_BASE_MODEL_PATH"
        )
    args.base_model = base

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"合并完成：{out}")
    print("推理时设置：FTKB_BACKEND=local FTKB_BASE_MODEL_PATH=%s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
