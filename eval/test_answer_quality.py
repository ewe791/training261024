"""F-E2 —— B 层：回答质量评测（对齐 4.1.1 ~ 4.1.5，共 9 项指标）。

  4.1.1 术语与格式适配   B1  术语准确率（规范术语 + 口径值一致，人工可接管）
                        B1F 格式遵循率（制度解释/产品说明/操作步骤/故障处理 四模板）
  4.1.2 忠实回答         B2  忠实度（spancheck，确定性）
                        B2C 关键条件完整性（上限/须经/逾期等条件是否遗漏）
                        B6  幻觉率（片段外事实，人工可接管）
  4.1.3 多片段整合       B3  整合完整率（每个片段的关键要点是否都被覆盖）
  4.1.4 引用与出处       B4  引用一致性（编号有效 + 被引句有证据支撑）
  4.1.5 资料不足拒答     B5  拒答及时率（无召回/弱相关/冲突 三类）
                        B5M 缺失信息指出率（说明缺什么 + 给下一步）

输出：eval/results_answer_quality.json、eval/results_answer_quality_detail.jsonl
运行：python eval/test_answer_quality.py [--role base|finetuned|both]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import config  # noqa: E402
from common.dataio import count_by, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from common.formatting import (  # noqa: E402
    check_format,
    load_formats,
    condition_coverage,
    fragment_coverage,
    strip_markers,
    term_check,
)
from common.llm import get_llm  # noqa: E402
from common.text import (  # noqa: E402
    citation_map,
    content_units,
    coverage,
    faithfulness,
    is_abstain,
)

B1_FILE = config.ANNO_DIR / "b1_terms.jsonl"
B6_FILE = config.ANNO_DIR / "b6_hallucination.jsonl"
MISSING_HINTS = ["缺少", "缺失", "未覆盖", "未包含", "未能覆盖", "信息不足"]
NEXT_HINTS = ["建议", "请联系", "联系", "确认", "补充", "上传"]



def _merge_detail(path, rows: list[dict], role: str) -> list[dict]:
    """明细文件按角色合并，不要整文件覆盖。

    覆盖会让后跑的角色把先跑的挤掉，compare.py 就找不到同一条样本的
    「基座 vs 微调」对照，改善示例直接变成 0 条。
    """
    old = []
    if path.exists():
        old = [r for r in read_jsonl(path) if r.get("role") != role]
    merged = old + rows
    write_jsonl(path, merged)
    return merged

def _load_annotations(path: Path) -> dict[str, int]:
    rows = read_jsonl(path)
    out: dict[str, int] = {}
    for r in rows:
        if r.get("label") in (0, 1):
            out[r["id"]] = int(r["label"])
    return out


def _emit_template(path: Path, ids: list[str]) -> None:
    if path.exists():
        return
    write_jsonl(path, [{"id": i, "label": None, "note": "1=正确/无幻觉，0=错误/有幻觉"} for i in ids])


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def evaluate(role: str, samples: list[dict]):
    llm = get_llm(role)
    b1_ann = _load_annotations(B1_FILE)
    b6_ann = _load_annotations(B6_FILE)

    acc = {
        "b1_hit": 0, "b1_n": 0,
        "b1f_hit": 0, "b1f_n": 0,
        "b1f_by_fmt": {},
        "b2_hit": 0, "b2_n": 0,
        "b2c_sum": 0.0, "b2c_n": 0,
        "b3_sum": 0.0, "b3_n": 0,
        "b4_sum": 0.0, "b4_n": 0,
        "b5_hit": 0, "b5_n": 0,
        "b5m_hit": 0, "b5m_n": 0,
        "b6_hit": 0, "b6_n": 0,
        "over_abstain": 0,
    }
    by_refuse_type: dict[str, list[int]] = {}
    details: list[dict] = []
    b1_src = "auto_rule"
    b6_src = "auto_rule"

    for s in samples:
        parsed = llm.run(s)
        reply = parsed.reply or parsed.raw
        ctx = s.get("contexts") or []
        ctx_texts = [f"{c.get('title', '')}\n{c['text']}" for c in ctx]
        ctx_map = {c["idx"]: c for c in ctx}
        exp = s["expected"]
        expect_abstain = bool(exp.get("should_abstain"))
        abstained = is_abstain(reply)
        counted = (not abstained) and (not expect_abstain)
        if abstained and not expect_abstain:
            acc["over_abstain"] += 1

        plain = strip_markers(reply)   # 忠实度/引用只在去格式标记后的正文上计算
        faith, sent_details = faithfulness(plain, ctx_texts)
        n_sent = len(sent_details)
        n_sup = sum(1 for d in sent_details if d["supported"])

        # B1F 分母用「所有应作答样本」，不排除被误判/真拒答的那些：
        # 该答却拒答，本身就是没按格式模板输出，且这样两侧分母才一致。
        fmt_ok = None
        if (not expect_abstain) and exp.get("format"):
            fmt_ok = check_format(reply, exp["format"])["pass"]
            acc["b1f_n"] += 1
            acc["b1f_hit"] += int(fmt_ok)
            slot = acc["b1f_by_fmt"].setdefault(exp["format"], {"hit": 0, "n": 0})
            slot["n"] += 1
            slot["hit"] += int(fmt_ok)

        terms_ok = None
        req_terms = exp.get("required_terms") or []
        if counted and req_terms:
            if s["id"] in b1_ann:
                terms_ok = b1_ann[s["id"]] == 1
                b1_src = "human_annotation"
                acc["b1_n"] += 1
                acc["b1_hit"] += int(terms_ok)
            else:
                # 术语实例级：标准答案里的每个规范术语都必须出现在回答中（别名替代同样算未命中）
                tc = term_check(reply, req_terms, ctx_texts)
                alias_bad = set(tc["alias_used"])
                hit = sum(1 for t in req_terms if t in reply and t not in alias_bad)
                acc["b1_n"] += len(req_terms)
                acc["b1_hit"] += hit
                terms_ok = hit == len(req_terms)

        conds_score = None
        if counted and exp.get("conditions"):
            conds_score, _ = condition_coverage(reply, exp["conditions"])
            acc["b2c_sum"] += conds_score
            acc["b2c_n"] += 1

        points_score = None
        groups = exp.get("point_groups")
        if counted and groups:
            points_score, _ = fragment_coverage(plain, groups)
            acc["b3_sum"] += points_score
            acc["b3_n"] += 1

        cite_score = None
        if counted and exp.get("must_cite"):
            cited = citation_map(plain)
            if not cited:
                cite_score = 0.0
            else:
                scores = []
                for item in cited:
                    ok_nums = [n for n in item["citations"] if n in ctx_map]
                    if not ok_nums:
                        scores.append(0.0)
                        continue
                    best = max(coverage(item["sentence"], content_units(ctx_map[n]["text"]))
                               for n in ok_nums)
                    scores.append(1.0 if best >= 0.80 else 0.0)
                cite_score = sum(scores) / len(scores)
            acc["b4_sum"] += cite_score
            acc["b4_n"] += 1

        if counted:
            acc["b2_hit"] += n_sup
            acc["b2_n"] += n_sent
            if s["id"] in b6_ann:
                halluc = b6_ann[s["id"]] == 0
                b6_src = "human_annotation"
            else:
                halluc = any(d["coverage"] < 0.60 for d in sent_details)
            acc["b6_hit"] += int(halluc)
            acc["b6_n"] += 1

        # 4.1.5 拒答：及时率 + 缺失信息指出率
        if expect_abstain:
            rtype = exp.get("refuse_type") or "unknown"
            acc["b5_n"] += 1
            acc["b5_hit"] += int(abstained)
            by_refuse_type.setdefault(rtype, [0, 0])
            by_refuse_type[rtype][0] += int(abstained)
            by_refuse_type[rtype][1] += 1
            # 分母是全部应拒答样本：没拒答就等于没说清缺什么
            acc["b5m_n"] += 1
            has_missing = any(h in reply for h in MISSING_HINTS)
            has_next = any(h in reply for h in NEXT_HINTS)
            acc["b5m_hit"] += int(abstained and has_missing and has_next)

        details.append({
            "id": s["id"],
            "role": role,
            "category": s["category"],
            "format": exp.get("format"),
            "refuse_type": exp.get("refuse_type"),
            "question": next((m["content"] for m in reversed(s["messages"])
                              if m["role"] == "user"), ""),
            "answer": reply,
            "output": parsed.raw,
            "faithfulness": round(faith, 4),
            "sentences": sent_details,
            "terms_ok": terms_ok,
            "format_ok": fmt_ok,
            "conditions_score": conds_score,
            "points_score": points_score,
            "citation_score": cite_score,
            "should_abstain": expect_abstain,
            "abstained": abstained,
            "missing_info_stated": (any(h in reply for h in MISSING_HINTS)
                                    and any(h in reply for h in NEXT_HINTS)),
            "contexts": [{"idx": c["idx"], "title": c["title"]} for c in ctx],
        })

    metrics = {
        "B1": _ratio(acc["b1_hit"], acc["b1_n"]),
        "B1F": _ratio(acc["b1f_hit"], acc["b1f_n"]),
        "B2": _ratio(acc["b2_hit"], acc["b2_n"]),
        "B2C": _ratio(acc["b2c_sum"], acc["b2c_n"]),
        "B3": _ratio(acc["b3_sum"], acc["b3_n"]),
        "B4": _ratio(acc["b4_sum"], acc["b4_n"]),
        "B5": _ratio(acc["b5_hit"], acc["b5_n"]),
        "B5M": _ratio(acc["b5m_hit"], acc["b5m_n"]),
        "B6": _ratio(acc["b6_hit"], acc["b6_n"]),
    }
    meta = {
        "B1_source": b1_src,
        "B6_source": b6_src,
        "n": {
            "B1": acc["b1_n"], "B1F": acc["b1f_n"], "B2_sentences": acc["b2_n"],
            "B2C": acc["b2c_n"], "B3": acc["b3_n"], "B4": acc["b4_n"],
            "B5": acc["b5_n"], "B5M": acc["b5m_n"], "B6": acc["b6_n"],
        },
        "over_abstain": acc["over_abstain"],
        "refuse_by_type": {k: {"hit": v[0], "n": v[1]} for k, v in by_refuse_type.items()},
        # 四类答案格式全覆盖登记：n=0 表示测试集没有这类样本，报告必须显式标注
        "format_by_type": {
            k: {"hit": acc["b1f_by_fmt"].get(k, {}).get("hit", 0),
                "n": acc["b1f_by_fmt"].get(k, {}).get("n", 0)}
            for k in load_formats()
        },
        "by_category": count_by(samples, "category"),
    }
    return metrics, details, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="both", choices=["base", "finetuned", "both"])
    args = ap.parse_args()

    samples = read_jsonl(config.QUALITY_TEST_PATH)
    if not samples:
        print("缺少 eval/quality_test.jsonl，请先运行 python eval/split_dataset.py")
        return 1

    config.ANNO_DIR.mkdir(parents=True, exist_ok=True)
    answerable = [s["id"] for s in samples if not s["expected"].get("should_abstain")]
    _emit_template(B1_FILE, answerable[:30])
    _emit_template(B6_FILE, answerable[:50])

    roles = ["base", "finetuned"] if args.role == "both" else [args.role]
    payload: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "backend": config.BACKEND,
        "models": {"base": config.BASE_MODEL_NAME, "finetuned": config.FINETUNED_MODEL_NAME},
        "testset": {"path": "eval/quality_test.jsonl", "total": len(samples),
                    "by_category": count_by(samples, "category")},
        "targets": {k: v for k, v in config.TARGETS.items() if k.startswith("B")},
    }
    all_details: list[dict] = []
    for role in roles:
        metrics, details, meta = evaluate(role, samples)
        payload[role] = metrics
        payload.setdefault("meta", {})[role] = meta
        all_details.extend(details)
        print(f"[{role}] " + "  ".join(f"{k}={v:.2%}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in metrics.items()))

    if args.role != "both" and config.RES_QUALITY_PATH.exists():
        old = read_json(config.RES_QUALITY_PATH, {}) or {}
        same_env = (old.get("backend") == payload["backend"]
                    and (old.get("testset") or {}).get("total") == payload["testset"]["total"])
        if not same_env:
            print(f"\n[注意] 已有结果来自 backend={old.get('backend')}、"
                  f"测试集 {(old.get('testset') or {}).get('total')} 条，与本次不一致，未合并。")
            print("       请把另一半角色也跑一遍（--role base / --role finetuned）后再 compare。")
        else:
            for r in ("base", "finetuned"):
                if r not in payload and r in old:
                    payload[r] = old[r]
            # meta 也必须合并：否则只跑一半角色时，另一侧的 meta
            # （format_by_type / refuse_by_type / 样本数）会整体丢失，
            # 报告里「四类格式是否都有测试样本」这类检查会误判为未通过。
            for r in ("base", "finetuned"):
                if r in old.get("meta", {}) and r not in payload.get("meta", {}):
                    payload.setdefault("meta", {})[r] = old["meta"][r]

    write_json(config.RES_QUALITY_PATH, payload)
    _merge_detail(config.RES_DETAIL_QUALITY_PATH, all_details, args.role)
    print(f"\n写出 eval/results_answer_quality.json 与 detail({len(all_details)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
