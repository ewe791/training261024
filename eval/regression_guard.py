"""F-E4 —— 回归护栏（微调后必跑）。

  R1 通用对话能力：10 条日常问答，微调后不得明显变差
  R2 拒答能力：越权拒答率 A6 与拒答及时率 B4 不得下降
  R3 格式遵循：工具调用输出 JSON 合法率 = 100%
  R4 无灾难性遗忘：同一测试集上微调总分不低于基座

判定：任一 R 项失败 → 标记"微调需回退调整"，不得直接交付。
输出：eval/results_regression.json
运行：python eval/regression_guard.py
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import config  # noqa: E402
from common.dataio import read_json, read_jsonl, write_json  # noqa: E402
from common.llm import get_llm  # noqa: E402
from common.protocol import make_answer, parse_output  # noqa: E402
from common.tools import system_prompt  # noqa: E402

DAILY_QUESTIONS = [
    "你好，介绍一下你能做什么",
    "帮我把这段话说得更简洁：本制度自发布之日起执行，未尽事宜由财务部负责解释。",
    "把「差旅住宿标准」按城市档次整理成三点",
    "如果同事问我报销发票要求，我该怎么转述",
    "「事前申请、事后报销」是什么意思",
    "帮我想三个跟差旅有关的常见问题",
    "用一句话总结公司考勤的基本要求",
    "如何向新员工解释数据权限的概念",
    "把「每晚上限 600 元」改写成正式的制度表述",
    "我需要准备哪些材料才能提交报销",
]

CHINESE = re.compile(r"[\u4e00-\u9fa5]")


def _quality_score(text: str) -> bool:
    """通用对话粗筛：非空、中文、可读、不是协议残片或空话。"""
    if not text or not text.strip():
        return False
    t = text.strip()
    if "{" in t or "}" in t:
        return False
    if not CHINESE.search(t):
        return False
    if len(t) < 15:
        return False
    if t.count("unknown") > 0:
        return False
    return True


def run_r1() -> dict:
    rows = []
    for role in ("base", "finetuned"):
        llm = get_llm(role)
        ok = 0
        samples = []
        for q in DAILY_QUESTIONS:
            messages = [
                {"role": "system", "content": system_prompt("kb_user")},
                {"role": "user", "content": q},
            ]
            text, native = llm.chat(messages, None)
            parsed = parse_output(text, native)
            passed = _quality_score(parsed.reply or parsed.raw)
            ok += int(passed)
            samples.append({"question": q, "output": (parsed.reply or parsed.raw)[:200],
                            "pass": passed})
        rows.append({"role": role, "rate": round(ok / len(DAILY_QUESTIONS), 4),
                     "samples": samples})
    base = next(r for r in rows if r["role"] == "base")
    ft = next(r for r in rows if r["role"] == "finetuned")
    passed = ft["rate"] >= base["rate"] - 0.10
    return {
        "id": "R1", "name": "通用对话能力",
        "base": base["rate"], "finetuned": ft["rate"],
        "pass": passed,
        "rule": "微调后不得明显变差（允许 ≤10 个百分点回落）",
        "detail": base["samples"] + ft["samples"],
    }


def run_r2(tool_res: dict, qual_res: dict) -> dict:
    a6b = (tool_res.get("base") or {}).get("A6")
    a6f = (tool_res.get("finetuned") or {}).get("A6")
    b5b = (qual_res.get("base") or {}).get("B5")
    b5f = (qual_res.get("finetuned") or {}).get("B5")
    b5mb = (qual_res.get("base") or {}).get("B5M")
    b5mf = (qual_res.get("finetuned") or {}).get("B5M")
    passed = (a6f is not None and a6b is not None and a6f >= a6b and a6f >= 1.0) and (
        b5f is not None and b5b is not None and b5f >= b5b) and (
        b5mf is not None and b5mb is not None and b5mf >= b5mb)
    return {
        "id": "R2", "name": "拒答能力",
        "base": a6b, "finetuned": a6f,
        "extra": {"B5_base": b5b, "B5_finetuned": b5f,
                  "B5M_base": b5mb, "B5M_finetuned": b5mf},
        "pass": passed,
        "rule": "A6 越权拒答 =100% 且不下降；B5 拒答及时率、B5M 缺失信息指出率不下降",
        "detail": [],
    }


def run_r3(details: list[dict]) -> dict:
    ft = [d for d in details if d["role"] == "finetuned"]
    invalid = [d for d in ft if not d["json_valid"]]
    rate = 1.0 if not ft else round(1 - len(invalid) / len(ft), 4)
    return {
        "id": "R3", "name": "格式遵循（JSON 合法率）",
        "base": None, "finetuned": rate,
        "pass": rate >= 1.0,
        "rule": "A 层工具调用输出 JSON 合法率必须 =100%",
        "detail": [{"id": d["id"], "output": d["output"][:120]} for d in invalid[:5]],
    }


def run_r4(tool_res: dict, qual_res: dict) -> dict:
    def total(res: dict) -> float:
        vals = []
        for k in ("A1", "A2", "A3", "A4", "A5", "A6"):
            v = (res.get("tool") or {}).get(k)
            if v is not None:
                vals.append(v)
        for k in ("B1", "B1F", "B2", "B2C", "B3", "B4", "B5", "B5M"):
            v = (res.get("qual") or {}).get(k)
            if v is not None:
                vals.append(v)
        v = (res.get("qual") or {}).get("B6")
        if v is not None:
            vals.append(1 - v)
        return sum(vals) / len(vals) if vals else 0.0

    base = {"tool": tool_res.get("base"), "qual": qual_res.get("base")}
    ft = {"tool": tool_res.get("finetuned"), "qual": qual_res.get("finetuned")}
    b, f = total(base), total(ft)
    return {
        "id": "R4", "name": "无灾难性遗忘（综合总分）",
        "base": round(b, 4), "finetuned": round(f, 4),
        "pass": f >= b,
        "rule": "同一测试集上微调综合分不得低于基座",
        "detail": [],
    }


def main() -> int:
    tool_res = read_json(config.RES_TOOL_PATH)
    qual_res = read_json(config.RES_QUALITY_PATH)
    if not tool_res or not qual_res:
        print("缺少评测结果，请先跑 A/B 两层评测")
        return 1
    tool_details = read_jsonl(config.RES_DETAIL_TOOL_PATH)

    items = [run_r1(), run_r2(tool_res, qual_res), run_r3(tool_details),
             run_r4(tool_res, qual_res)]
    failed = [i["id"] for i in items if not i["pass"]]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "backend": config.BACKEND,
        "items": items,
        "failed": failed,
        "conclusion": "PASS" if not failed else "FAIL",
        "verdict": "回归护栏全通过，可交付" if not failed else "微调需回退调整，不得直接交付",
    }
    write_json(config.RES_REGRESSION_PATH, payload)

    print("回归护栏：")
    for i in items:
        print(f"  {i['id']} {i['name']}: base={i['base']} ft={i['finetuned']} "
              f"→ {'PASS' if i['pass'] else 'FAIL'}（{i['rule']}）")
    print("\n结论：" + payload["verdict"])
    print("写出 eval/results_regression.json")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
