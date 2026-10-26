"""4.1.1~4.1.5 的确定性判定工具：答案格式、关键条件、多片段要点、冲突召回、术语口径。

对应关系：
  4.1.1 术语与格式  → check_format / term_check
  4.1.2 忠实回答    → condition_coverage（关键条件是否遗漏）
  4.1.3 多片段整合  → point_coverage（每个片段的关键要点是否都被覆盖）
  4.1.4 引用标注    → 见 eval/test_answer_quality.py（引用编号 + 证据一致）
  4.1.5 资料不足    → detect_conflict / 拒答分型
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from common.config import CONFIG_DIR
from common.text import (
    CITATION_RE,
    claim_tokens,
    content_units,
    coverage,
    norm,
    split_sentences,
)

STEP_RE = re.compile(r"(?:^|\n)\s*(?:[1-9]\d?[.、)）]|[①②③④⑤⑥⑦⑧⑨⑩])")
CONDITION_RE = re.compile(
    r"不超过|不得超过|不得|须|须经|应|应当|需提前|仅|逾期|原则上|必须|不低于|上限|下限|超出"
)


@lru_cache(maxsize=None)
def load_formats() -> dict:
    return json.loads((CONFIG_DIR / "formats.json").read_text(encoding="utf-8"))["formats"]


@lru_cache(maxsize=None)
def load_terms() -> list[dict]:
    return json.loads((CONFIG_DIR / "terms.json").read_text(encoding="utf-8"))["terms"]


FAULT_HINTS = ("常见问题", "为什么", "看不到", "报错", "失败", "怎么办",
               "无法", "不能", "异常", "超时", "异常处理", "排查", "故障")
STEP_HINTS = ("流程", "步骤", "导入", "审批", "申请", "打卡", "查询", "如何", "操作指引")
PRODUCT_DOC_HINTS = ("手册", "说明书", "产品文档", "用户指南", "操作指南")
PRODUCT_SEC_HINTS = ("概述", "功能", "权限说明", "使用", "手册", "特性", "模块", "简介")
POLICY_DOC_HINTS = ("制度", "政策", "管理办法", "规定", "条例", "准则", "规范")


def pick_format(section_title: str, doc: str = "") -> str:
    """按文档类型 + 章节标题选择答案格式（故障处理 > 操作步骤 > 产品说明 > 制度解释）。

    标题优先；标题无信号时**必须看文档类型**——否则产品手册的普通章节
    （如「2.1 登录」）会被默认判成制度解释，把标签教错。
    """
    t = section_title or ""
    d = doc or ""
    if any(k in t for k in FAULT_HINTS):
        return "故障处理"
    if any(k in t for k in STEP_HINTS):
        return "操作步骤"
    if any(k in t for k in PRODUCT_SEC_HINTS):
        return "产品说明"
    if any(k in d for k in PRODUCT_DOC_HINTS):
        return "产品说明"
    if any(k in d for k in POLICY_DOC_HINTS):
        return "制度解释"
    return "制度解释"


def _clean_point(s: str) -> str:
    s = re.sub(r"^\s*(?:[1-9]\d?[.、)）]|[①②③④⑤⑥⑦⑧⑨⑩])\s*", "", s)
    return s.strip().rstrip("。；;")


def build_formatted(fmt: str, points: list[str], cites: list[int],
                    topic: str = "") -> str:
    """按格式模板拼装标准答案（训练/评测共用同一模板，保证口径一致）。"""
    pts = [_clean_point(p) for p in points if _clean_point(p)]
    cite = "".join(f"[{i}]" for i in cites) or "[1]"
    if fmt == "操作步骤":
        n = min(len(pts), 4)
        lines = [f"{i + 1}. {p}" for i, p in enumerate(pts[:n])]
        if not lines:
            lines = ["1. 按制度要求提交申请"]
        lines[-1] = lines[-1] + cite
        return "\n".join(lines)
    if fmt == "产品说明":
        a = pts[0] if pts else "该功能的具体说明请见片段"
        b = pts[1] if len(pts) > 1 else a
        c = pts[2] if len(pts) > 2 else ""
        out = f"【功能】{a}\n【使用方式】{b}"
        if c:
            out += f"\n【限制】{c}"
        return out + cite
    if fmt == "故障处理":
        # 现象优先用问题本身；【原因】只在原文真有原因句时才写。
        # 否则会把「解决措施」硬贴上【原因】标签，教模型编造因果。
        symptom = topic or (pts[0] if pts else "出现该现象")
        cause = next((p for p in pts if CAUSE_RE.search(p)), "")
        steps = [p for p in pts if p != cause] or pts or ["按上述步骤处理"]
        out = f"【现象】{symptom}"
        if cause:
            out += f"\n【原因】{cause}"
        return out + f"\n【处理】{steps[0]}{cite}"
    # 制度解释（默认）：【依据】容纳多条，避免关键条件句被槽位挤掉
    a = pts[0] if pts else "按制度执行"
    b = "；".join(pts[1:]) if len(pts) > 1 else a
    return f"【结论】{a}\n【依据】{b}{cite}"


def check_format(reply: str, fmt: str) -> dict:
    spec = load_formats().get(fmt)
    if not spec:
        return {"pass": True, "missing": [], "rule": "无格式要求"}
    text = reply or ""
    missing: list[str] = []
    if spec.get("ordered"):
        steps = len(STEP_RE.findall("\n" + text))
        ok = steps >= int(spec.get("min_steps", 2))
        return {
            "pass": ok,
            "missing": [] if ok else [f"分步≥{spec.get('min_steps', 2)}"],
            "rule": "操作步骤：至少 2 个有序步骤",
            "steps": steps,
        }
    pos = -1
    for marker in spec["required"]:
        idx = text.find(marker)
        if idx < 0 or idx < pos:
            missing.append(marker)
        else:
            pos = idx
    return {
        "pass": not missing,
        "missing": missing,
        "rule": f"{fmt}：按顺序包含 {'、'.join(spec['required'])}",
    }


def condition_sentences(text: str, limit: int = 4) -> list[str]:
    """抽取含约束条件的关键句（上限/须经/仅/逾期…），用于 4.1.2 关键条件完整性。"""
    out = []
    for s in split_sentences(CITATION_RE.sub("", text or "")):
        s = s.strip()
        if s.startswith("|"):      # 表格行不作为条件句，避免行间差异造成噪声
            continue
        if len(s) >= 8 and CONDITION_RE.search(s):
            out.append(s)
    return out[:limit]


def condition_coverage(answer: str, conditions: list[str], threshold: float = 0.6) -> tuple[float, list[dict]]:
    """每个关键条件是否在答案中留有踪迹（内容词覆盖率 ≥ threshold）。"""
    if not conditions:
        return 1.0, []
    ans_units = content_units(answer)
    details = []
    hit = 0
    for c in conditions:
        cov = coverage(c, ans_units)
        ok = cov >= threshold
        hit += int(ok)
        details.append({"condition": c, "coverage": round(cov, 3), "covered": ok})
    return hit / len(conditions), details


def point_coverage(answer: str, points: list[str], threshold: float = 0.6) -> tuple[float, list[dict]]:
    """4.1.3：多片段场景下，每个片段的关键要点是否被覆盖。"""
    if not points:
        return 1.0, []
    ans_units = content_units(answer)
    details = []
    hit = 0
    for p in points:
        cov = coverage(p, ans_units)
        ok = cov >= threshold
        hit += int(ok)
        details.append({"point": p, "coverage": round(cov, 3), "covered": ok})
    return hit / len(points), details


def fragment_coverage(answer: str, groups: list[list[str]],
                      threshold: float = 0.6) -> tuple[float, list[dict]]:
    """4.1.3 片段级覆盖：每个召回片段至少要有一条关键信息被用上。

    groups 为「每个片段的关键句候选」，命中任一候选即视为该片段被整合。
    """
    ans_units = content_units(answer)
    if not groups:
        return 1.0, []
    details = []
    hit = 0
    for g in groups:
        best = 0.0
        best_sent = ""
        for p in g:
            cov = coverage(p, ans_units)
            if cov > best:
                best, best_sent = cov, p
        ok = best >= threshold
        hit += int(ok)
        details.append({"point": short(best_sent, 40), "coverage": round(best, 3), "covered": ok})
    return hit / len(groups), details


def detect_conflict(texts: list[str], topic_overlap: float = 0.45) -> dict:
    """4.1.5 冲突召回：同一主题（内容高度重叠）但关键口径（金额/时长/比例）不一致。"""
    if len(texts) < 2:
        return {"conflict": False}
    a, b = texts[0], texts[1]
    ua, ub = content_units(a), content_units(b)
    if not ua or not ub:
        return {"conflict": False}
    overlap = len(ua & ub) / max(1, min(len(ua), len(ub)))
    ca, cb = claim_tokens(a), claim_tokens(b)
    conflict = overlap >= topic_overlap and bool(ca) and bool(cb) and set(ca) != set(cb)
    return {
        "conflict": conflict,
        "topic_overlap": round(overlap, 3),
        "claims_a": sorted(set(ca)),
        "claims_b": sorted(set(cb)),
    }


def term_check(reply: str, required_terms: list[str], context_texts: list[str]) -> dict:
    """4.1.1 术语：必须使用规范术语，且不得只用口语别名替代。"""
    text = reply or ""
    ctx = norm(" ".join(context_texts))
    missing = [t for t in required_terms if t not in text]
    alias_used = []
    for item in load_terms():
        canon = item["canonical"]
        if canon not in required_terms:
            continue
        for alias in item.get("aliases", []):
            if alias in text and canon not in text:
                alias_used.append(alias)
    claims = claim_tokens(text)
    bad_claims = [c for c in claims if c not in ctx]
    ok = not missing and not alias_used and not bad_claims
    return {
        "pass": ok,
        "missing_terms": missing,
        "alias_used": alias_used,
        "claims_outside_context": bad_claims,
    }


CAUSE_RE = re.compile(r"(?:主要)?原因|由于|因为|所致|导致|造成")

MARKER_RE = re.compile(r"【[^】]{1,10}】")
STEP_PREFIX_RE = re.compile(r"(?m)^\s*(?:[1-9]\d?[.、)）]|[①②③④⑤⑥⑦⑧⑨⑩])\s*")
UNIT_IN_HEADER_RE = re.compile(r"[（(]([^）)]{1,6})[）)]")


def _table_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def table_points(text: str, limit: int = 8) -> list[str]:
    """把 Markdown 表格逐行转成自然语言陈述句。

    表格是知识库的高频内容（住宿标准、假期额度…），如果直接按行当句子用，
    模型会学到输出「| 一线城市 | 北京 | 600 |」这种残片。这里转成：
        一线城市（北京、上海、广州、深圳）：每晚上限 600 元
    """
    out: list[str] = []
    lines = [ln for ln in (text or "").splitlines()]
    i = 0
    while i < len(lines) and len(out) < limit:
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue
        block = []
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            block.append(lines[i])
            i += 1
        if len(block) < 3:
            continue
        header = _table_cells(block[0])
        for row in block[2:]:            # 跳过表头与 --- 分隔行
            cells = _table_cells(row)
            if not cells or not cells[0]:
                continue
            key = cells[0]
            tail_h = header[-1] if header else ""
            mid = [c for c in cells[1:-1] if c]
            last = cells[-1] if len(cells) > 1 else ""

            # 中间列：短枚举用括号点缀，长说明放进正文（用逗号并列）
            mid_enum, mid_body = "", ""
            if mid:
                joined = "，".join(mid)
                if "、" in joined or len(joined) <= 24:
                    mid_enum = f"（{joined}）"
                else:
                    mid_body = joined

            unit_m = UNIT_IN_HEADER_RE.search(tail_h)
            core_h = UNIT_IN_HEADER_RE.sub("", tail_h).strip()
            parts: list[str] = []
            if mid_body:
                parts.append(mid_body)
            if last:
                if core_h in ("说明", "备注", "其他", ""):
                    parts.append(last)
                else:
                    unit = unit_m.group(1) if unit_m else ""
                    num = last if (not unit or last.endswith(unit)) else f"{last} {unit}"
                    parts.append(f"{core_h} {num}".strip())
            body = "，".join(p for p in parts if p)
            sent = f"{key}{mid_enum}：{body}" if body else f"{key}{mid_enum}"
            out.append(sent)
    return out


def strip_markers(text: str) -> str:
    """去掉【结论】/1. 这类格式标记，避免格式词被算成"片段外内容"。"""
    t = MARKER_RE.sub("", text or "")
    t = STEP_PREFIX_RE.sub("", t)
    return t


def short(text: str, n: int = 90) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def section_path() -> Path:
    return CONFIG_DIR
