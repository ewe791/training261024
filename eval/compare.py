"""F-E3 —— 前后对比与改善示例。

读取 results_tool_calling.json / results_answer_quality.json 与两份明细，
产出：
  - 对比表数据（基座 vs 微调，含变化与是否达标）
  - 改善示例 ≥10 条（基座失败 / 微调成功）
  - 失败案例 ≥5 条（微调仍失败 + 原因分析 + 改进建议）

输出：eval/compare.json
运行：python eval/compare.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import config  # noqa: E402
from common.dataio import read_json, read_jsonl, write_json  # noqa: E402

# (指标名, 判定方式, 目标展示, 能力分组)
METRIC_META = {
    "A1": ("A1 工具选择准确率", "max", "≥90%", "A 层 · 工具调用"),
    "A2": ("A2 参数抽取 F1", "max", "≥85%", "A 层 · 工具调用"),
    "A3": ("A3 缺参澄清率", "max", "≥90%", "A 层 · 工具调用"),
    "A4": ("A4 多轮指代正确率", "max", "≥85%", "A 层 · 工具调用"),
    "A5": ("A5 写操作确认率", "max", "≥90%", "A 层 · 工具调用"),
    "A6": ("A6 越权拒答率（硬指标）", "hard", "=100%", "A 层 · 工具调用"),
    "B1": ("B1 术语准确率", "max", "≥90%", "4.1.1 术语与格式"),
    "B1F": ("B1F 格式遵循率", "max", "≥90%", "4.1.1 术语与格式"),
    "B2": ("B2 忠实度", "max", "≥95%", "4.1.2 忠实回答"),
    "B2C": ("B2C 关键条件完整性", "max", "≥90%", "4.1.2 忠实回答"),
    "B6": ("B6 幻觉率", "min", "≤5%", "4.1.2 忠实回答"),
    "B3": ("B3 多片段整合完整率", "max", "≥90%", "4.1.3 多片段整合"),
    "B4": ("B4 引用一致性", "max", "≥95%", "4.1.4 引用与出处"),
    "B5": ("B5 拒答及时率", "max", "≥90%", "4.1.5 资料不足拒答"),
    "B5M": ("B5M 缺失信息指出率", "max", "≥90%", "4.1.5 资料不足拒答"),
}

CATEGORY_CAUSE = {
    "参数抽取边界问题": "训练集中同类实体（编号相邻、场景相近）样本不足，模型在长尾取值上仍会扰动字符",
    "场景相似度高导致混淆": "T4 与 T1 的场景描述高度重合，边界样本配比偏低，模型退化为高频工具",
    "训练数据覆盖不足": "该场景在训练集内变体过少（<3 种表述），模型未见过的问法直接放弃",
    "拒答策略未固化为强约束": "越权/无依据场景的拒答样本以话术为主，缺少带上下文干扰的负例",
}


def _cause(detail: dict) -> tuple[str, str]:
    reason = detail.get("reason") or ""
    cat = detail.get("category")
    if "参数" in reason:
        cause = "参数抽取边界问题"
        advice = "补 60–100 条同实体近邻取值（编号相邻/仅一位差异）的对照样本，并在数据构造时做最小编辑扰动"
    elif cat == "T4":
        cause = "场景相似度高导致混淆"
        advice = f"围绕期望工具 {detail.get('expected', {}).get('tool_name', '')} 补 40 条与混淆工具成对出现的对比样本"
    elif cat in ("T2", "T5", "T6"):
        cause = "训练数据覆盖不足"
        advice = "把该类场景的训练变体从 2 个扩到 6–8 个，覆盖口语化简称与带干扰信息的问法"
    else:
        cause = "训练数据覆盖不足"
        advice = "补足该场景变体，并在 System Prompt 中显式给出该工具的触发条件"
    return cause, advice


def build_compare_table(tool_res: dict, qual_res: dict) -> list[dict]:
    rows = []
    for key, (label, kind, target_txt, group) in METRIC_META.items():
        src = tool_res if key.startswith("A") else qual_res
        base = (src.get("base") or {}).get(key)
        ft = (src.get("finetuned") or {}).get(key)
        if base is None or ft is None:
            continue
        delta = round(ft - base, 4)
        tgt = config.TARGETS[key]
        ok = (ft >= tgt - 1e-9) if kind != "min" else (ft <= tgt + 1e-9)
        rows.append({
            "metric": key, "label": label, "base": base, "finetuned": ft,
            "delta": delta, "target_text": target_txt, "target": tgt,
            "pass": ok, "hard": kind == "hard", "group": group,
        })
    return rows


def improvements(tool_details: list[dict], qual_details: list[dict],
                 limit: int = 10, a_cap: int = 5) -> list[dict]:
    base_by_id = {d["id"]: d for d in tool_details if d["role"] == "base"}
    ft_by_id = {d["id"]: d for d in tool_details if d["role"] == "finetuned"}
    picks: list[dict] = []
    per_cat: dict[str, int] = defaultdict(int)
    for sid, b in base_by_id.items():
        if len(picks) >= a_cap:
            break
        f = ft_by_id.get(sid)
        if not f or b["correct"] or not f["correct"]:
            continue
        if per_cat[b["category"]] >= 2:
            continue
        per_cat[b["category"]] += 1
        picks.append({
            "source": "A 层",
            "category": b["category"],
            "question": b["question"],
            "base_output": b["output"],
            "ft_output": f["output"],
            "reason": f"基座：{b['reason']}；微调：{f['reason']}",
        })
    picks.sort(key=lambda x: x["category"])

    qbase = {d["id"]: d for d in qual_details if d["role"] == "base"}
    qft = {d["id"]: d for d in qual_details if d["role"] == "finetuned"}
    for sid, b in qbase.items():
        if len(picks) >= limit:
            break
        f = qft.get(sid)
        if not f or f["should_abstain"]:
            continue
        base_bad = (not b["abstained"]) and (
            (b["format_ok"] is False)
            or (b["faithfulness"] < 0.95)
            or ((b["citation_score"] or 1) < 0.9)
            or ((b["conditions_score"] or 1) < 0.9))
        ft_ok = (f["format_ok"] is not False) and f["faithfulness"] >= 0.95 and (
            (f["citation_score"] or 1) >= 0.9) and ((f["conditions_score"] or 1) >= 0.9)
        if base_bad and ft_ok:
            picks.append({
                "source": "B 层",
                "category": b["category"],
                "question": b["question"],
                "base_output": b["answer"],
                "ft_output": f["answer"],
                "reason": (f"基座：格式{'未遵循' if b['format_ok'] is False else '遵循'}、"
                           f"忠实度 {b['faithfulness']:.2f}、引用 {(b['citation_score'] or 0):.2f}；"
                           f"微调：格式{'遵循' if f['format_ok'] else '未遵循'}、"
                           f"忠实度 {f['faithfulness']:.2f}、引用 {(f['citation_score'] or 1):.2f}"),
            })
    return picks[:limit]


def failures(tool_details: list[dict], qual_details: list[dict], limit: int = 6) -> list[dict]:
    out: list[dict] = []
    seen_q: set[str] = set()
    cause_count: dict[tuple, int] = defaultdict(int)
    ft = [d for d in tool_details if d["role"] == "finetuned" and not d["correct"]]
    a_fail_cap = max(limit - 3, 3)
    for d in ft:
        if len(out) >= a_fail_cap:
            break
        if d["question"] in seen_q:
            continue
        cause, advice = _cause(d)
        key = (d.get("expected", {}).get("tool_name"), cause)
        if cause_count[key] >= 2:
            continue
        cause_count[key] += 1
        seen_q.add(d["question"])
        cause, advice = _cause(d)
        out.append({
            "source": "A 层", "category": d["category"], "question": d["question"],
            "expected": d["expected"], "ft_output": d["output"],
            "reason": d["reason"], "cause": cause, "advice": advice,
        })
    qft = [d for d in qual_details if d["role"] == "finetuned"]
    for d in qft:
        if len(out) >= limit:
            break
        if d["should_abstain"]:
            continue
        if (d["format_ok"] is False) or d["faithfulness"] < 0.95 or (
                (d["citation_score"] or 1) < 0.9) or ((d["conditions_score"] or 1) < 0.9):
            if d["format_ok"] is False:
                cause = "输出格式模板未固化为默认行为"
                advice = ("把四类模板写成训练样本的固定骨架（每题都按模板生成标准答案），"
                          "并补 40 条「同一问题、四种格式」的对照样本")
            elif (d["conditions_score"] or 1) < 0.9:
                cause = "关键条件句提取不足"
                advice = "在标准答案中显式保留上限/须经/逾期类条件句，并补 30 条「多条件并列」样本"
            elif (d["citation_score"] or 1) < 0.9:
                cause = "引用标注习惯未固化"
                advice = "训练样本统一在末尾标注 [n]，并补 20 条「跨片段引用」样本"
            else:
                cause = "训练数据覆盖不足"
                advice = "补 30 条多片段整合样本，并对含表格/编号的片段做引用标注增强"
            out.append({
                "source": "B 层", "category": d["category"], "question": d["question"],
                "expected": "答案句子均被片段支撑，且引用编号正确",
                "ft_output": d["answer"],
                "reason": (f"格式{'未遵循' if d['format_ok'] is False else '遵循'}，"
                           f"忠实度 {d['faithfulness']:.2f}，条件完整性 "
                           f"{(d['conditions_score'] if d['conditions_score'] is not None else 1):.2f}"),
                "cause": cause,
                "advice": advice,
            })
    return out


def main() -> int:
    tool_res = read_json(config.RES_TOOL_PATH)
    qual_res = read_json(config.RES_QUALITY_PATH)
    if not tool_res or not qual_res:
        print("缺少评测结果，请先运行 eval/test_tool_calling.py 与 eval/test_answer_quality.py")
        return 1
    tool_details = read_jsonl(config.RES_DETAIL_TOOL_PATH)
    qual_details = read_jsonl(config.RES_DETAIL_QUALITY_PATH)

    table = build_compare_table(tool_res, qual_res)
    imps = improvements(tool_details, qual_details, limit=10)
    fails = failures(tool_details, qual_details, limit=6)

    payload = {
        "generated_at": tool_res.get("generated_at"),
        "backend": tool_res.get("backend"),
        "table": table,
        "improvements": imps,
        "failures": fails,
        "hard_metric_failed": [r["metric"] for r in table if r["hard"] and not r["pass"]],
        "failed_metrics": [r["metric"] for r in table if not r["pass"]],
    }
    write_json(config.COMPARE_PATH, payload)

    print("指标        基座      微调      变化     是否达标")
    for r in table:
        print(f"{r['label'][:12]:<12} {r['base']:>7.2%} {r['finetuned']:>7.2%} "
              f"{r['delta']:>+8.2%}   {'✓' if r['pass'] else '✗'}")
    print(f"\n改善示例 {len(imps)} 条，失败案例 {len(fails)} 条")
    print("写出 eval/compare.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
