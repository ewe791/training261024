"""一键跑完整流程：数据构造 → 划分 → A/B 评测 → 对比 → 回归护栏 → 报告。

运行：
  python scripts/run_all.py                     # 默认 stub 后端（流水线自测）
  python scripts/run_all.py --backend openai    # 真实模型（需 .env 配置）
  python scripts/run_all.py --skip-build        # 复用已有数据，只重跑评测
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def step(title: str, args: list[str], env: dict) -> None:
    print("\n" + "=" * 72)
    print(f"▶ {title}")
    print("=" * 72)
    r = subprocess.run([PY] + args, cwd=str(ROOT), env=env)
    if r.returncode not in (0,):
        raise SystemExit(f"步骤失败：{' '.join(args)} (exit {r.returncode})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=os.getenv("FTKB_BACKEND", "stub"),
                    choices=["stub", "openai", "local"])
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-train", action="store_true", help="仅评测时使用")
    args = ap.parse_args()

    env = dict(os.environ)
    env["FTKB_BACKEND"] = args.backend
    env.setdefault("PYTHONIOENCODING", "utf-8")

    if not args.skip_build:
        step("F1 构造训练数据", [str(ROOT / "scripts" / "build_dataset.py")], env)
        step("F-E0 划分训练/测试集（85/15，场景不重叠）",
             [str(ROOT / "eval" / "split_dataset.py")], env)
    step("F2 训练数据体检（dry-run）", [str(ROOT / "train" / "finetune_lora.py"), "--dry-run"], env)

    if args.backend == "local":
        # 本地/云端单卡：基座与微调分两个进程跑，避免两个 7B 模型同时驻留显存
        print("\n[local 后端] 基座与微调将分进程评测（各自跑完即释放显存）")
        for role in ("base", "finetuned"):
            step(f"F-E1 A 层评测 · {role}",
                 [str(ROOT / "eval" / "test_tool_calling.py"), "--role", role], env)
            step(f"F-E2 B 层评测 · {role}",
                 [str(ROOT / "eval" / "test_answer_quality.py"), "--role", role], env)
    else:
        step("F-E1 A 层：工具调用能力评测",
             [str(ROOT / "eval" / "test_tool_calling.py")], env)
        step("F-E2 B 层：回答质量评测",
             [str(ROOT / "eval" / "test_answer_quality.py")], env)

    step("F-E3 前后对比与改善示例", [str(ROOT / "eval" / "compare.py")], env)
    step("F-E4 回归护栏", [str(ROOT / "eval" / "regression_guard.py")], env)
    step("F-E5 生成评测报告", [str(ROOT / "scripts" / "gen_report.py")], env)

    print("\n全部完成 → reports/eval_finetune.md")
    if args.backend == "stub":
        print("注意：本次为 stub 模拟数据，仅验证流水线；真实结论请用 --backend openai/local 重跑。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
