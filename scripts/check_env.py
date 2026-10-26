"""上云/换机后的环境自检：一条命令看清「能不能开训、还差什么」。

运行：
    python scripts/check_env.py
    python scripts/check_env.py --base-model /root/autodl-tmp/Qwen2.5-7B-Instruct
    python scripts/check_env.py --no-card      # 无卡模式：GPU 检查降级为提示

退出码：0 = 可以开训；1 = 有阻塞项（BLOCK）；2 = 仅告警（WARN）。

为什么要 --no-card：无卡模式下 torch.cuda.is_available() 必然是 False，
如果不区分，脚本会把「正常的无卡状态」报成阻塞项，让人以为环境坏了。
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import config  # noqa: E402

OK, WARN, BLOCK = "OK  ", "WARN", "BLOCK"
rows: list[tuple[str, str, str]] = []

# 无卡模式（实例只有 CPU）下，GPU 不可用是预期状态，不算阻塞
NO_CARD = False


def add(level: str, item: str, detail: str) -> None:
    rows.append((level, item, detail))


def check_python() -> None:
    v = sys.version_info
    add(OK if v >= (3, 10) else BLOCK, "Python 版本",
        f"{platform.python_version()}（{platform.system()} {platform.machine()}）")
    add(OK, "解释器路径", sys.executable)


def check_torch() -> bool:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        add(BLOCK, "PyTorch", f"未安装或导入失败：{exc}")
        add(WARN, "PyTorch 排查",
            "若在 venv 里：多半是创建时没加 --system-site-packages，隔离掉了镜像自带的 "
            "CUDA 版 torch → rm -rf .venv && bash scripts/cloud_setup.sh")
        return False
    cuda_ok = torch.cuda.is_available()
    # torch 装好就是 OK，「用不了 GPU」是 GPU 这一项的问题，分开报才不会误导
    add(OK, "PyTorch",
        f"{torch.__version__} / CUDA 编译版本 {torch.version.cuda or '无'}")
    if not cuda_ok:
        add(WARN if NO_CARD else BLOCK, "GPU",
            "无卡模式：未分配 GPU，属预期状态（有卡开机后重跑本脚本即可）"
            if NO_CARD else
            "torch.cuda.is_available() = False（驱动/镜像不对，或选了 CPU 实例）")
        return True
    n = torch.cuda.device_count()
    add(OK, "GPU 数量", str(n))
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
        cap = torch.cuda.get_device_capability(i)
        bf16 = "支持 bf16" if cap[0] >= 8 else "仅 fp16（训练脚本会自动切）"
        add(OK if mem >= 15 else WARN, f"GPU[{i}]", f"{name} / {mem:.1f} GB / {bf16}")
        if mem < 15:
            hint = ("8–14G：7B 只能走 4bit（config/lora_sft.yaml 里 load_in_4bit: true），"
                    "且很紧张；建议本机先用 config/lora_sft_1.5b.yaml 跑通全链路")
            if mem < 10:
                hint = ("<10G：7B 训练基本不可行（4bit 也只剩 1–2G 余量）；"
                        "本机请用 config/lora_sft_1.5b.yaml 验证流程，7B 上云")
            add(WARN, f"GPU[{i}] 显存提示", hint)
    free, total = torch.cuda.mem_get_info(0)
    add(OK if free > 8 * 1024 ** 3 else WARN, "显存占用",
        f"空闲 {free / 1024 ** 3:.1f} GB / 共 {total / 1024 ** 3:.1f} GB")
    return True


def check_gpu_smi() -> None:
    """没有 torch 时，也能从 nvidia-smi 看到卡型号与显存（装机前先判断卡够不够）。"""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:  # noqa: BLE001
        add(WARN, "nvidia-smi", "无法执行（没有 NVIDIA 驱动或未在 PATH）")
        return
    if out.returncode != 0 or not out.stdout.strip():
        add(WARN, "nvidia-smi", "未检测到 NVIDIA GPU（只有核显/AMD 的话无法本机训练）")
        return
    for i, line in enumerate(out.stdout.strip().splitlines()):
        parts = [x.strip() for x in line.split(",")]
        name = parts[0] if parts else "?"
        mem_txt = parts[1] if len(parts) > 1 else "?"
        drv = parts[2] if len(parts) > 2 else "?"
        try:
            mem_gb = float(mem_txt.lower().replace("mib", "")) / 1024
        except ValueError:
            mem_gb = 0.0
        add(OK, f"GPU[{i}]（未装 torch，来自 nvidia-smi）", f"{name} / {mem_gb:.1f} GB / 驱动 {drv}")
        if 0 < mem_gb < 10:
            add(WARN, f"GPU[{i}] 显存提示",
                "<10G：7B 训练基本不可行（4bit 也只剩 1–2G 余量）；"
                "本机请用 config/lora_sft_1.5b.yaml 验证全链路，7B 上云")
        elif mem_gb < 15:
            add(WARN, f"GPU[{i}] 显存提示",
                "8–14G：7B 只能走 4bit（config/lora_sft.yaml 里 load_in_4bit: true），且很紧张")


def check_deps() -> None:
    need = {
        "transformers": "训练与推理",
        "peft": "LoRA",
        "datasets": "训练数据集",
        "accelerate": "device_map / Trainer",
        "sentencepiece": "Qwen tokenizer",
    }
    optional = {"modelscope": "国内下载基座模型", "openai": "在线模型后端", "dotenv": ".env 解析"}
    for mod, why in need.items():
        try:
            m = importlib.import_module(mod)
            add(OK, f"依赖 {mod}", f"{getattr(m, '__version__', '?')}（{why}）")
        except Exception:  # noqa: BLE001
            add(BLOCK, f"依赖 {mod}", f"缺失（{why}）→ pip install {mod}")
    for mod, why in optional.items():
        try:
            m = importlib.import_module(mod)
            add(OK, f"可选 {mod}", f"{getattr(m, '__version__', '?')}（{why}）")
        except Exception:  # noqa: BLE001
            add(WARN, f"可选 {mod}", f"未安装（{why}），不影响核心流程")


def check_model(path: str | None) -> None:
    target = path or os.getenv("FTKB_BASE_MODEL_PATH", "")
    if not target:
        add(WARN, "基座模型路径", "未设置 FTKB_BASE_MODEL_PATH，也未传 --base-model")
        return
    p = Path(target)
    if not p.exists():
        add(BLOCK, "基座模型路径", f"{target} 不存在（模型还没下载完？）")
        return
    need_files = ["config.json"]
    got = {f.name for f in p.iterdir()} if p.is_dir() else set()
    missing = [f for f in need_files if f not in got]
    weights = [f.name for f in p.glob("*.safetensors")] if p.is_dir() else []
    size_gb = sum(f.stat().st_size for f in p.glob("*")) / 1024 ** 3 if p.is_dir() else 0
    if missing:
        add(BLOCK, "基座模型文件", f"缺少 {missing}")
    elif not weights:
        add(BLOCK, "基座模型文件", "未找到 *.safetensors（可能下载中断）")
    else:
        add(OK, "基座模型", f"{target}｜{len(weights)} 个权重分片｜{size_gb:.1f} GB")
    tok = [f for f in ("tokenizer.json", "tokenizer_config.json") if f in got]
    add(OK if tok else WARN, "tokenizer", "、".join(tok) if tok else "缺少 tokenizer 文件")

    adapter = os.getenv("FTKB_ADAPTER_PATH", "")
    if adapter:
        ap = Path(adapter)
        ok = (ap / "adapter_config.json").exists()
        add(OK if ok else WARN, "LoRA 适配器",
            f"{adapter}（{'已就绪' if ok else '还没训练或路径不对'}）")


def check_data() -> None:
    pairs = [
        (config.TRAIN_PATH, "A 层训练集"),
        (config.QUALITY_TRAIN_PATH, "B 层训练集"),
        (config.TEST_PATH, "A 层测试集"),
        (config.QUALITY_TEST_PATH, "B 层测试集"),
    ]
    for path, name in pairs:
        if path.exists():
            n = sum(1 for _ in path.open(encoding="utf-8"))
            add(OK if n > 0 else BLOCK, name, f"{path.relative_to(ROOT)}｜{n} 条")
        else:
            add(WARN, name, f"{path.relative_to(ROOT)} 不存在 → 先跑 build_dataset + split_dataset")


def check_disk() -> None:
    du = shutil.disk_usage(ROOT)
    free_gb = du.free / 1024 ** 3
    add(OK if free_gb > 40 else WARN, "磁盘剩余",
        f"{free_gb:.0f} GB（7B 模型约 15G + 权重输出若干 G）")


def check_runtime() -> None:
    add(OK if config.BACKEND in ("stub", "local", "openai") else WARN,
        "推理后端 FTKB_BACKEND", config.BACKEND)
    add(OK, "PyYAML 配置", str((config.CONFIG_DIR / "lora_sft.yaml").exists()))
    if (ROOT / ".env").exists():
        add(OK, ".env", f"已加载（{ROOT / '.env'}）")
    else:
        add(WARN, ".env", "未找到，将使用环境变量/默认值（cp .env.example .env）")
    add(OK if os.getenv("PYTHONIOENCODING") else WARN, "PYTHONIOENCODING",
        os.getenv("PYTHONIOENCODING", "未设置，中文日志可能乱码 → export PYTHONIOENCODING=utf-8"))


def main() -> int:
    global NO_CARD
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--no-card", action="store_true",
                    help="无卡模式：没有 GPU 属预期，GPU 相关项降级为提示")
    args = ap.parse_args()
    NO_CARD = bool(args.no_card or os.getenv("FTKB_NO_CARD") == "1")

    check_python()
    has_torch = check_torch()
    if not has_torch:
        check_gpu_smi()
    if NO_CARD:
        add(WARN, "运行模式", "无卡模式（只能装环境/造数据/看日志，不能训练）")
    check_deps()
    check_model(args.base_model)
    check_data()
    check_disk()
    check_runtime()

    width = max(len(r[1]) for r in rows) + 2
    print("=" * (width + 70))
    print("环境自检报告".center(width + 60))
    print("=" * (width + 70))
    for level, item, detail in rows:
        print(f"[{level}] {item:<{width}} {detail}")
    print("=" * (width + 70))

    blocks = [r for r in rows if r[0] == BLOCK]
    warns = [r for r in rows if r[0] == WARN]
    if blocks:
        print(f"结论：{len(blocks)} 项阻塞、{len(warns)} 项告警 → 先修阻塞项再训练")
        for _l, item, detail in blocks:
            print(f"  · {item}: {detail}")
        return 1
    if not has_torch:
        return 1
    if NO_CARD:
        print(f"结论：环境已就绪（无卡模式，GPU 相关项已跳过；{len(warns)} 项提示）")
        print("      有卡开机后执行：python scripts/check_env.py   再确认一次，然后即可开训")
        return 0
    print(f"结论：可以开训（{len(warns)} 项告警，不影响主流程）")
    return 0 if not warns else 2


if __name__ == "__main__":
    raise SystemExit(main())
