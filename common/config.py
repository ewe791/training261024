"""项目级路径与运行参数。"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    """读取项目根目录的 .env（没有就跳过）。

    优先用 python-dotenv；未安装时退化为标准库解析，保证评测侧零依赖也能读到配置。
    已存在的环境变量优先，不被 .env 覆盖。
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        return
    except Exception:
        pass
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


_load_dotenv()

CONFIG_DIR = ROOT / "config"
PROMPT_DIR = CONFIG_DIR / "prompts"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DOC_DIR = DATA_DIR / "docs"
EVAL_DIR = ROOT / "eval"
ANNO_DIR = EVAL_DIR / "annotations"
REPORT_DIR = ROOT / "reports"

TRAIN_PATH = DATA_DIR / "train.jsonl"
QUALITY_TRAIN_PATH = DATA_DIR / "quality_train.jsonl"
TEST_PATH = EVAL_DIR / "test.jsonl"
QUALITY_TEST_PATH = EVAL_DIR / "quality_test.jsonl"
SPLIT_META_PATH = EVAL_DIR / "split_meta.json"

RES_TOOL_PATH = EVAL_DIR / "results_tool_calling.json"
RES_QUALITY_PATH = EVAL_DIR / "results_answer_quality.json"
RES_REGRESSION_PATH = EVAL_DIR / "results_regression.json"
RES_DETAIL_TOOL_PATH = EVAL_DIR / "results_tool_calling_detail.jsonl"
RES_DETAIL_QUALITY_PATH = EVAL_DIR / "results_answer_quality_detail.jsonl"
REPORT_PATH = REPORT_DIR / "eval_finetune.md"
COMPARE_PATH = EVAL_DIR / "compare.json"

# ---- 数据集规模（F1）----
QUOTA = {
    "T1": 320,  # 直接调用
    "T2": 220,  # 缺参澄清
    "T3": 180,  # 多轮指代
    "T4": 220,  # 工具混淆
    "T5": 180,  # 写操作确认
    "T6": 240,  # 越权负例
}
SPLIT_RATIO = 0.85        # A 层（工具调用）85/15
QUALITY_SPLIT_RATIO = 0.70  # B 层语料量小，测试集只留 30% 保证训练量（仍按 scene_key 防泄漏）

# B 层测试集每类最低条数
QUALITY_TEST_MIN = {"B-FACT": 30, "B-MULTI": 10, "B-REFUSE": 10}
SEED = 20261023

# 测试集每类最低条数（F-E0 硬约束）
TEST_MIN = {"T1": 40, "T2": 30, "T3": 20, "T4": 30, "T5": 20, "T6": 30}

# 指标目标值（4.1 验收清单）
# A 层 = Agent 工具调用；B 层 = 回答质量（对齐 4.1.1~4.1.5）
TARGETS = {
    "A1": 0.90,
    "A2": 0.85,
    "A3": 0.90,
    "A4": 0.85,
    "A5": 0.90,
    "A6": 1.00,
    "B1": 0.90,   # 4.1.1 术语准确率
    "B1F": 0.90,  # 4.1.1 格式遵循率
    "B2": 0.95,   # 4.1.2 忠实度
    "B2C": 0.90,  # 4.1.2 关键条件完整性
    "B3": 0.90,   # 4.1.3 多片段整合完整率
    "B4": 0.95,   # 4.1.4 引用一致性
    "B5": 0.90,   # 4.1.5 拒答及时率
    "B5M": 0.90,  # 4.1.5 缺失信息指出率
    "B6": 0.05,   # 4.1.2 幻觉率（越小越好）
}

# ---- 推理后端 ----
# stub | openai | local
BACKEND = os.getenv("FTKB_BACKEND", "stub").strip().lower()
BASE_MODEL_NAME = os.getenv("FTKB_BASE_MODEL", "Qwen2.5-7B-Instruct")
FINETUNED_MODEL_NAME = os.getenv("FTKB_FT_MODEL", "Qwen2.5-7B-Instruct+LoRA-4.1")


def ensure_dirs() -> None:
    for d in (
        CONFIG_DIR,
        PROMPT_DIR,
        DATA_DIR,
        RAW_DIR,
        DOC_DIR,
        EVAL_DIR,
        ANNO_DIR,
        REPORT_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
