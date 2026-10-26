"""推理后端：stub（流水线自测模拟器）/ openai（在线）/ local（本地 LoRA）。

统一接口：
    text, native_tool_calls = llm.chat(messages, tools)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from typing import Any

from common import config
from common.protocol import (
    Parsed,
    dumps,
    make_answer,
    make_clarify,
    make_refuse,
    make_tool_call,
    parse_output,
)
from common.tools import load_tools, required_params
from common.text import claim_tokens, content_units, coverage, split_sentences
from common.formatting import (
    build_formatted,
    condition_sentences,
    detect_conflict,
    load_terms,
    table_points,
)

# --------------------------------------------------------------------------------------
# 用户话语中的实体抽取（三种后端共用，用于 stub 与参数兜底）
# --------------------------------------------------------------------------------------
PATTERNS = {
    "user_id": r"\bU\d{3,4}\b",
    "order_id": r"\bO\d{4}\b",
    "assessment_id": r"\bA\d{3,4}\b",
    "rule_code": r"\bR\d{2,3}\b",
    "case_id": r"\bC\d{3,4}\b",
    "unit_id": r"\bKU-[0-9]{14}-[0-9A-F]{4}\b",
    "faq_id": r"\bF\d{3,4}\b",
    "file_paths": r"seed/docs/[^\s，,。；;]+\.(?:md|docx|pdf|txt)",
}
EVENT_TYPE_HINTS = [
    ("order_create", ["下单", "创建订单", "新订单", "下单场景"]),
    ("order_pay", ["支付", "付款", "支付场景"]),
    ("after_sale_apply", ["售后", "退款申请", "仅退款", "售后场景"]),
    ("logistics_complaint", ["物流投诉", "快递投诉", "物流场景", "投诉物流"]),
]
DEPARTMENTS = ["财务部", "市场部", "HR部"]
ROLES = ["财务专员", "市场专员", "HR专员"]
PERSONS = ["张三", "李四", "王五"]
OVERREACH_HINTS = [
    "别管权限", "不管权限", "不用管权限", "绕过", "我是管理员", "现在授权",
    "原文", "全文", "导出", "所有人", "全部用户", "别人", "私密", "工资",
    "不给全文就给摘要", "直接给我", "越权", "偷偷",
]
WRITE_HINTS = ["停用", "启用", "删除", "删掉", "审核通过", "驳回", "配权限", "授权",
               "发布", "导入", "加入黑名单", "移出黑名单", "改状态", "归档",
               "只给", "改成", "改为", "加进黑名单", "拉黑", "打了回", "打回", "判通过"]
CONFIRM_HINTS = ["确认", "就这条", "是的", "对，就是"]
FAQ_HINT = ["faq", "常见问题", "问答推荐"]


def extract_entities(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field, pat in PATTERNS.items():
        m = re.findall(pat, text or "")
        if m:
            out[field] = m if field == "file_paths" else m[0]
    for name, kws in EVENT_TYPE_HINTS:
        if any(k in (text or "") for k in kws):
            out["event_type"] = name
            break
    for name in DEPARTMENTS:
        if name in (text or ""):
            out.setdefault("_departments", []).append(name)
    for name in ROLES:
        if name in (text or ""):
            out.setdefault("_roles", []).append(name)
    for name in PERSONS:
        if name in (text or ""):
            out.setdefault("_persons", []).append(name)
    if "全局" in (text or "") or "所有人" in (text or ""):
        out.setdefault("_global", True)
    return out


def _hash_unit(*parts: str) -> float:
    h = hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


# --------------------------------------------------------------------------------------
# Stub 模拟器
# --------------------------------------------------------------------------------------
SKILL = {
    "base": {
        "tool_pick": 0.76, "param": 0.62, "clarify": 0.34, "multiturn": 0.46,
        "confirm": 0.40, "refuse": 0.36, "json": 0.93,
        # B 层（回答质量）
        "format": 0.42, "point": 0.50, "cite": 0.55, "faith": 0.78, "cond": 0.25,
        "term": 0.45,
        "refuse_no": 0.55, "refuse_weak": 0.50, "refuse_conflict": 0.30,
    },
    "finetuned": {
        "tool_pick": 0.96, "param": 0.92, "clarify": 0.94, "multiturn": 0.88,
        "confirm": 0.95, "refuse": 1.00, "json": 1.00,
        # B 层（回答质量）
        "format": 0.97, "point": 0.96, "cite": 0.99, "faith": 0.98, "cond": 0.99,
        "term": 0.97,
        "refuse_no": 0.98, "refuse_weak": 0.97, "refuse_conflict": 0.98,
    },
}

GENERIC_UNITS = {
    "规定", "制度", "标准", "要求", "什么", "多少", "公司", "是否", "可以", "需要",
    "如何", "哪些", "这个", "那个", "一下", "关于", "相关", "分别", "怎么", "一个",
}

INTENT_RULES = {
    "risk": [
        ("toggle_rule", ["停用", "禁用", "关掉", "启用规则", "打开规则", "启用"]),
        ("review_case", ["审核案件", "审核通过", "判通过", "驳回", "审一下", "审核意见"]),
        ("get_dashboard", ["看板", "检查量", "命中率", "分布", "建案", "趋势"]),
        ("get_assessment", ["评估"]),
        ("risk_check", ["风险检查", "查风险", "有风险", "有没有风险", "有没有问题",
                        "风控一下", "风险等级", "跑个风险", "风控过"]),
        ("get_user_profile", ["画像", "退款率", "历史订单", "黑名单状态", "注册多久"]),
        ("manage_blacklist", ["黑名单", "拉黑"]),
        ("list_cases", ["案件", "待审", "案子"]),
        ("list_rules", ["规则", "策略"]),
    ],
    "kb_admin": [
        ("import_knowledge", ["导入", "上传"]),
        ("delete_units", ["删除", "删掉", "清理"]),
        ("review_faq", ["审核faq", "发布faq", "驳回faq", "审核 FAQ", "发布 FAQ",
                        "发布", "打回", "通过"]),
        ("set_unit_permissions", ["权限", "只给", "可见", "授权"]),
        ("list_faq_recommendations", ["faq", "常见问题", "问答推荐", "推荐列表"]),
        ("get_unit", ["讲了什么", "正文", "详情", "这一份", "这个单元"]),
        ("list_units", ["知识单元", "有哪些文档", "找一下", "搜一下", "文档列表"]),
        ("list_knowledge_gaps", ["没答上来", "知识缺口", "答不上", "缺口"]),
        ("get_dashboard_metrics", ["看板", "访问量", "token", "指标"]),
        ("update_unit", ["改标题", "改状态", "归档", "更新"]),
    ],
}


CONFUSION_TRAPS = {
    "risk_check": "get_assessment",
    "get_assessment": "risk_check",
    "list_rules": "toggle_rule",
    "toggle_rule": "list_rules",
    "list_cases": "review_case",
    "review_case": "list_cases",
    "get_user_profile": "manage_blacklist",
    "list_units": "get_unit",
    "get_unit": "list_units",
    "delete_units": "update_unit",
    "review_faq": "list_faq_recommendations",
    "list_faq_recommendations": "review_faq",
}


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _history_text(messages: list[dict]) -> str:
    """对话历史（不含 system，避免把 Prompt 里的示例实体当成上文）。"""
    return " ".join(
        (m.get("content") or "") for m in messages[:-1] if m.get("role") != "system"
    )


def _pick_intent(text: str, agent: str) -> str | None:
    rules = INTENT_RULES.get(agent, [])
    low = (text or "").lower()
    for name, kws in rules:
        if any(k.lower() in low for k in kws):
            return name
    return None


def _args_for(tool: str, ent: dict[str, Any], text: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if tool == "risk_check":
        for k in ("event_type", "user_id", "order_id"):
            if k in ent:
                args[k] = ent[k]
    elif tool in ("get_assessment",):
        if "assessment_id" in ent:
            args["assessment_id"] = ent["assessment_id"]
    elif tool in ("toggle_rule",):
        if "rule_code" in ent:
            args["rule_code"] = ent["rule_code"]
        args["action"] = "disable" if ("停用" in text or "关闭" in text) else "enable"
    elif tool == "review_case":
        if "case_id" in ent:
            args["case_id"] = ent["case_id"]
        args["review_result"] = "reject" if "驳回" in text else "pass"
        args["review_remark"] = "按风控策略驳回" if "驳回" in text else "资料齐全，予以通过"
    elif tool in ("get_user_profile", "list_cases"):
        if "user_id" in ent:
            args["user_id"] = ent["user_id"]
    elif tool == "manage_blacklist":
        if "移除" in text or "移出" in text:
            args["action"] = "remove"
        elif "加入" in text or "拉黑" in text:
            args["action"] = "add"
        else:
            args["action"] = "query"
        if "user_id" in ent:
            args["user_id"] = ent["user_id"]
        if args["action"] == "add":
            args["reason"] = "疑似团伙骗赔"
    elif tool == "get_dashboard":
        if "昨天" in text:
            args["range"] = "yesterday"
        elif "7" in text or "近一周" in text:
            args["range"] = "7d"
        elif "30" in text or "近一月" in text:
            args["range"] = "30d"
        else:
            args["range"] = "today"
    elif tool == "import_knowledge":
        if "file_paths" in ent:
            args["file_paths"] = ent["file_paths"]
    elif tool in ("get_unit", "delete_units", "update_unit", "set_unit_permissions"):
        if "unit_id" in ent:
            args["unit_id"] = ent["unit_id"] if tool != "delete_units" else None
        if tool == "delete_units":
            args["unit_ids"] = [ent["unit_id"]] if "unit_id" in ent else []
        if tool == "set_unit_permissions":
            targets = []
            for d in ent.get("_departments", []):
                targets.append({"target_type": "department", "target_value": d})
            for r in ent.get("_roles", []):
                targets.append({"target_type": "role", "target_value": r})
            for p in ent.get("_persons", []):
                targets.append({"target_type": "user", "target_value": p})
            if ent.get("_global"):
                targets.append({"target_type": "global", "target_value": ""})
            args["targets"] = targets
    elif tool == "review_faq":
        if "faq_id" in ent:
            args["faq_id"] = ent["faq_id"]
        args["action"] = "reject" if ("驳回" in text or "不通过" in text) else "approve"
    elif tool == "list_rules":
        if "rule_code" in ent:
            args["keyword"] = ent["rule_code"]
    elif tool == "get_dashboard_metrics":
        args = {}
    elif tool == "list_units":
        kw = re.search(r"[「“\"]([^」”\"]{2,8})[」”\"]", text)
        if kw:
            args["keyword"] = kw.group(1)
    elif tool == "list_faq_recommendations":
        args = {}
    return {k: v for k, v in args.items() if v not in (None, [], "")}


def _render_fragments(system: str) -> list[dict]:
    frags = []
    block = system.split("【知识片段】", 1)
    if len(block) < 2:
        return frags
    body = block[1]
    for m in re.finditer(
        r"\[(\d+)\]\s*(.+?)（(.+?)）\n(.*?)(?=\n\[\d+\]\s|\n【输出格式要求】|\Z)",
        body, re.S,
    ):
        frags.append({
            "idx": int(m.group(1)),
            "title": m.group(2).strip(),
            "unit_id": m.group(3).strip(),
            "text": m.group(4).strip(),
        })
    return frags


FMT_RE = re.compile(r"《(.+?)》模板")


def _render_format(system: str) -> str | None:
    m = FMT_RE.search(system or "")
    return m.group(1) if m else None


def _refuse_no_recall() -> str:
    return ("知识库中暂未找到与该问题相关的内容，无法作答。\n"
            "缺少的信息：该主题的制度条款或操作说明尚未入库。\n"
            "建议：联系知识管理员补充对应文档，或补充具体业务场景后再次提问。")


def _refuse_weak() -> str:
    return ("检索到的资料与该问题仅部分相关，未覆盖所问信息，无法据此作答。\n"
            "缺少的信息：该问题的具体标准与适用条件。\n"
            "建议：补充该主题的制度文件，或向对应部门确认后再作答。")


def _refuse_conflict(frags: list[dict]) -> str:
    a = sorted(set(claim_tokens(frags[0]["text"])))[:3]
    b = sorted(set(claim_tokens(frags[1]["text"])))[:3]
    v1 = "、".join(a) or "原口径"
    v2 = "、".join(b) or "新版口径"
    return (f"两份资料对同一事项的口径不一致（{v1} 与 {v2}），无法确定以哪一版为准，"
            "暂不给出结论。\n"
            "缺少的信息：该口径的现行有效版本或最新发文。\n"
            "建议：向制度归口部门确认版本后，再据有效版本作答。")


_TERMS_CACHE: list[str] | None = None


def _all_terms() -> list[str]:
    global _TERMS_CACHE
    if _TERMS_CACHE is None:
        _TERMS_CACHE = [t["canonical"] for t in load_terms()]
    return _TERMS_CACHE


def _sent_score(sent: str, q_units: set[str], first: bool) -> float:
    """句子打分：问题相关度为主，带口径数字与规范术语的条款句加权。"""
    score = len(content_units(sent) & q_units) * 10.0
    if claim_tokens(sent):
        score += 2.0
    # 含规范术语的句子优先：4.1.1 要求回答回到"规范术语"，而不是只答数字
    if any(t in sent for t in _all_terms()):
        score += 2.5
    if first:
        score += 1.0
    return score


def _pick_points(frags: list[dict], q_units: set[str], skill: dict, rng: float,
                 multi: bool) -> tuple[list[str], list[int]]:
    """按问题相关度挑要点；多片段场景每个片段各挑一条（4.1.3 的整合要求）。

    表格行不进候选，改用 table_points 转述后的句子——否则会输出
    「| 一线城市 | 北京 | 600 |」这种残片，术语和口径全丢。
    """
    per_frag: list[tuple[float, int, str]] = []
    all_cands: list[tuple[float, int, str]] = []
    for f in frags:
        cands: list[tuple[float, int, str]] = []
        for i, raw in enumerate(split_sentences(f["text"])):
            s = raw.strip()
            if len(s) < 8 or "|" in s:
                continue
            if s.endswith(("如下：", "如下:", "如下", "如下所示：", "以下：")):
                continue          # 引导句，实质内容在表格里
            cands.append((_sent_score(s, q_units, i == 0), f["idx"], s))
        # 表格转述句大幅加权：表格几乎总是口径核心（住宿标准、假期额度），
        # 只给 +3 会被"住宿标准每年度复核一次"这类泛泛条款挤掉，
        # 导致答案只有原则没有数字。真实模型也会优先引用带数值的那一行。
        for s in table_points(f["text"]):
            cands.append((_sent_score(s, q_units, False) + 10.0, f["idx"], s))
        if not cands:
            continue
        cands.sort(key=lambda x: (-x[0], x[1]))
        per_frag.append(cands[0])
        all_cands.extend(cands)

    if multi and len(per_frag) >= 2 and rng < skill["point"]:
        scored = per_frag[:2]
    else:
        all_cands.sort(key=lambda x: (-x[0], x[1]))
        scored = all_cands[:2]

    picked = [s for _sc, _i, s in scored]
    used = [i for _sc, i, _s in scored]

    # 关键条件句（4.1.2 不得遗漏）：按覆盖率判断哪些还没被带上
    conds = condition_sentences(" ".join(f["text"] for f in frags))
    ans_units = content_units(" ".join(picked))
    missing = [c for c in conds if coverage(c, ans_units) < 0.8][:2]
    if not multi and missing and rng < skill["cond"]:
        picked = picked + missing
        for c in missing:
            for f in frags:
                if c in f["text"] and f["idx"] not in used:
                    used.append(f["idx"])
                    break
    return picked, (used or [1])

def _stub_answer_quality(messages: list[dict], role: str, qid: str) -> str:
    skill = SKILL[role]
    system = messages[0].get("content") or ""
    question = _last_user(messages)
    frags = _render_fragments(system)
    fmt = _render_format(system)
    rng = _hash_unit(role, qid, question)

    def hit(p: float) -> bool:
        return rng < p

    # 1) 冲突召回（4.1.5）：主题相同但口径不一致
    if len(frags) >= 2:
        cf = detect_conflict([frags[0]["text"], frags[1]["text"]])
        if cf["conflict"]:
            if hit(skill["refuse_conflict"]):
                return make_answer(_refuse_conflict(frags))
            # 基座典型失败：无视冲突，只答片段[1]
            frags = frags[:1]

    # 2) 无召回（4.1.5）
    if not frags:
        if hit(skill["refuse_no"]):
            return make_answer(_refuse_no_recall())
        return make_answer("通常按公司现行做法执行，具体以最新通知为准。")

    q_units = content_units(question) - GENERIC_UNITS
    best = 0
    for f in frags:
        for s in split_sentences(f["text"]):
            best = max(best, len(content_units(s) & q_units))

    # 3) 弱相关 / 资料不足（4.1.5）
    if best == 0:
        if hit(skill["refuse_weak"]):
            return make_answer(_refuse_weak())
        seed = split_sentences(frags[0]["text"])
        return make_answer((seed[0] if seed else "按制度执行") + "。[1]")

    # 4) 正常作答：按 4.1.1 的格式模板组织
    picked, used = _pick_points(frags, q_units, skill, rng, multi=len(frags) >= 2)

    # 4.1.1 术语口径：用户用规范术语提问时复述该术语；用口语别名提问时
    # 回到规范术语（这是「行业术语适配」要考的能力，训练样本必须体现）。
    # 片段标题也算可用信息：术语可能只出现在标题里（如「请假类型与额度」）
    hay = " ".join(f.get("title", "") + " " + f["text"] for f in frags)
    topic_term = ""
    term_m = re.search('[\u300c\u300e]([^\u300d\u300f]{2,12})[\u300d\u300f]', question)
    if term_m and term_m.group(1) in hay:
        topic_term = term_m.group(1)
    if not topic_term:
        for t in _all_terms():
            if t in question and t in hay:
                topic_term = t
                break
    if not topic_term:
        for t in load_terms():
            if t["canonical"] in hay and any(
                a and a in question for a in (t.get("aliases") or [])
            ):
                topic_term = t["canonical"]
                break
    if topic_term and rng < skill["term"] and not any(topic_term in p for p in picked):
        picked = [f"{topic_term}：{picked[0]}"] + picked[1:]

    if not hit(skill["faith"]):
        picked = picked + ["另有补充口径：超标部分统一按 1200 元标准报销，无需事前审批"]

    cites = used if hit(skill["cite"]) else []
    if fmt and hit(skill["format"]):
        reply = build_formatted(fmt, picked, cites)
    else:
        body = "；".join(p for p in picked) + "。"
        reply = body + "".join(f"[{i}]" for i in cites)
    return make_answer(reply)


CONFIRM_FOR = {
    "toggle_rule": "list_rules",
    "review_case": "list_cases",
    "manage_blacklist": "manage_blacklist",
    "delete_units": "list_units",
    "set_unit_permissions": "get_unit",
    "update_unit": "get_unit",
    "review_faq": "list_faq_recommendations",
}
FABRICATED = {
    "order_id": "O2001", "user_id": "U1001", "event_type": "order_create",
    "assessment_id": "A001", "rule_code": "R05", "case_id": "C001",
    "unit_id": "KU-20260101000001-A1B2", "faq_id": "F001",
    "file_paths": ["seed/docs/财务报销制度.md"],
    "review_result": "pass", "review_remark": "已核对",
    "target_type": "department", "unit_ids": ["KU-20260101000001-A1B2"],
    "rule_codes": ["R05"],
}


def _corrupt(value: Any) -> Any:
    if isinstance(value, list):
        return value + ["tmp_extra"] if value else ["U1001"]
    if not isinstance(value, str):
        return value
    digits = re.findall(r"\d+", value)
    if digits:
        last = digits[-1]
        return value.replace(last, last[:-1] + str((int(last[-1]) + 1) % 10), 1)
    return value + "_x"


def _confirm_args(confirm: str, ent: dict, text: str) -> dict:
    if confirm == "list_rules":
        return {"keyword": ent["rule_code"]} if "rule_code" in ent else {}
    if confirm == "list_cases":
        return {}
    if confirm == "manage_blacklist":
        return {"action": "query", "user_id": ent.get("user_id", "")}
    if confirm == "list_units":
        return {"keyword": ent["unit_id"]} if "unit_id" in ent else {}
    if confirm == "get_unit":
        return {"unit_id": ent["unit_id"]} if "unit_id" in ent else {}
    return {}


def stub_chat(messages: list[dict], tools: list[dict] | None, role: str) -> tuple[str, list | None]:
    skill = SKILL[role]
    text = _last_user(messages)
    hist = _history_text(messages)
    tool_names = [t["function"]["name"] for t in (tools or [])]
    agent = "kb_user"
    if "risk_check" in tool_names:
        agent = "risk"
    elif "import_knowledge" in tool_names:
        agent = "kb_admin"

    qid = hashlib.md5(("|".join(m.get("content", "") for m in messages)).encode()).hexdigest()[:8]
    rng = _hash_unit(role, qid, text)
    rnd = random.Random(int(rng * 1_000_000))

    def hit(p: float) -> bool:
        return rng < p

    overreach = any(h in text for h in OVERREACH_HINTS)
    follow_confirm = any(h in text for h in CONFIRM_HINTS)
    is_write = any(h in text for h in WRITE_HINTS)

    # 1) 越权请求：先拒答（A6 硬指标）
    if overreach and not follow_confirm:
        if hit(skill["refuse"]):
            return make_refuse(
                "该操作超出您的权限范围，我无法执行。如需访问，请联系知识管理员开通对应权限。"
            ), None
        if agent == "kb_user":
            return make_answer(
                "按制度原文：一线城市差旅住宿每晚上限 600 元，二线城市 450 元，"
                "其他城市 350 元，凭发票实报实销。[1]"
            ), None
        if agent == "kb_admin":
            return make_tool_call("delete_units", {"unit_ids": FABRICATED["unit_ids"]}), None
        return make_tool_call("risk_check", {
            "event_type": "order_create", "user_id": "U1001", "order_id": "O2001"
        }), None

    if agent == "kb_user":
        return _stub_answer_quality(messages, role, qid), None

    ent = extract_entities(text + " " + hist)
    if hist and not hit(skill["multiturn"]):
        ent = extract_entities(text)  # 忽略上下文指代（基座典型缺陷）
    intent = _pick_intent(text, agent)
    if intent is None:
        intent = tool_names[0] if hit(skill["tool_pick"]) else rnd.choice(tool_names)
    elif not hit(skill["tool_pick"]):
        intent = CONFUSION_TRAPS.get(intent, intent)

    args = _args_for(intent, ent, text)
    req = required_params(agent, intent)
    missing = [r for r in req if r not in args]

    # 2) 缺参：追问 or 臆造（写在确认之前：参数不齐无法确认）
    if missing:
        multi_turn = bool(hist) and any(
            k in text for k in ("刚才", "这个", "那条", "该", "上一个", "另一个"))
        p_clarify = max(skill["clarify"], skill["multiturn"]) if multi_turn else skill["clarify"]
        if hit(p_clarify):
            return make_clarify(missing, f"还需要您提供{'、'.join(missing)}，我才能继续处理。"), None
        for m in missing:
            args[m] = FABRICATED.get(m, "unknown")

    # 3) 写操作确认
    if intent in CONFIRM_FOR and is_write and not follow_confirm:
        if hit(skill["confirm"]):
            confirm = CONFIRM_FOR[intent]
            return make_tool_call(confirm, _confirm_args(confirm, ent, text)), None

    # 4) 参数抽取噪声
    if args and not hit(skill["param"]):
        key = rnd.choice(sorted(args))
        args[key] = _corrupt(args[key])

    # 5) JSON 合法性
    if not hit(skill["json"]):
        return '{"type":"tool_call","name":"%s","arguments":{%s' % (intent, dumps(args)[1:-1]), None

    return make_tool_call(intent, args), None


# --------------------------------------------------------------------------------------
# OpenAI 兼容后端
# --------------------------------------------------------------------------------------
class OpenAILLM:
    def __init__(self, model: str):
        from openai import OpenAI  # 延迟导入

        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        self.temperature = float(os.getenv("FTKB_TEMPERATURE", "0"))

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[str, list | None]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None)
        return (msg.content or ""), calls


# --------------------------------------------------------------------------------------
# 本地 transformers + peft 后端
# --------------------------------------------------------------------------------------
class LocalLLM:
    """本地 transformers + peft 后端。

    环境变量：
      FTKB_BASE_MODEL_PATH   基座模型目录
      FTKB_ADAPTER_PATH      LoRA 适配器目录（仅微调角色）
      FTKB_LOAD_4BIT=1       4bit 量化加载（16G 及以下显存评测时用）
      FTKB_MAX_NEW_TOKENS    生成上限，默认 512
    """

    def __init__(self, model_path: str, adapter_path: str | None = None):
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = int(os.getenv("FTKB_MAX_NEW_TOKENS", "512"))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side="left"
        )
        if os.getenv("FTKB_LOAD_4BIT") == "1":
            from transformers import BitsAndBytesConfig

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[str, list | None]:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0], None


# --------------------------------------------------------------------------------------
# 工厂
# --------------------------------------------------------------------------------------
class LLM:
    def __init__(self, role: str):
        self.role = role
        self.name = config.BASE_MODEL_NAME if role == "base" else config.FINETUNED_MODEL_NAME
        self.backend = config.BACKEND
        self._impl = self._build()

    def _build(self):
        if self.backend == "openai":
            model = (
                config.BASE_MODEL_NAME if self.role == "base"
                else os.getenv("FTKB_FT_MODEL_ID", config.FINETUNED_MODEL_NAME)
            )
            return OpenAILLM(model)
        if self.backend == "local":
            base_path = os.getenv("FTKB_BASE_MODEL_PATH", "").strip()
            if not base_path:
                raise RuntimeError(
                    "FTKB_BACKEND=local 但未设置 FTKB_BASE_MODEL_PATH。"
                    "请在 .env 里指向基座模型目录，例如 "
                    "FTKB_BASE_MODEL_PATH=/root/autodl-tmp/Qwen2.5-7B-Instruct"
                )
            adapter = None if self.role == "base" else (os.getenv("FTKB_ADAPTER_PATH") or None)
            if self.role != "base" and not adapter:
                raise RuntimeError(
                    "评测微调角色需要 FTKB_ADAPTER_PATH 指向 LoRA 适配器目录"
                    "（如 outputs/lora-knowledge-agent-4.1）"
                )
            return LocalLLM(base_path, adapter)
        return None  # stub

    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        if self._impl is None:
            return stub_chat(messages, tools, self.role)
        return self._impl.chat(messages, tools)

    def run(self, sample: dict) -> Parsed:
        tools = load_tools(sample["agent"])
        text, native = self.chat(sample["messages"], tools or None)
        return parse_output(text, native)


def get_llm(role: str) -> LLM:
    return LLM(role)
