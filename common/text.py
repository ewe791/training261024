"""中文文本处理与确定性指标工具（不依赖第三方库）。

忠实度沿用项目二 K-E3 的 spancheck 思路：
  answer 按句拆分 → 每句内容词在 context 中的覆盖率 ≥ 阈值即视为被支撑。
"""
from __future__ import annotations

import re

SENT_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")
CJK_RUN_RE = re.compile(r"[\u4e00-\u9fa5]+")
ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]*|\d+(?:\.\d+)?")
CITATION_RE = re.compile(r"\[(\d{1,2})\]")

STOPWORDS = set(
    "的 了 和 与 及 或 是 在 对 于 为 以 按 由 须 需 应 可 要 不 也 都 就 而 并 等 中 上 下 内 外 "
    "我们 你们 他们 本 该 其 此 这 那 有 无 未 已 将 把 被 从 到 至 后 前 时 年 月 日 个 条 项 "
    "如 若 请 你 我 他 她 它 一 二 三 四 五 六 七 八 九 十 之 所 者 者 呢 吗 啊 吧".split()
)

FIELD_KEYWORDS: dict[str, list[str]] = {
    "order_id": ["订单", "单号", "order"],
    "user_id": ["用户", "账号", "哪位", "user"],
    "event_type": ["场景", "事件", "类型", "下单", "支付", "售后", "物流"],
    "unit_id": ["知识单元", "哪一份", "文档", "单元"],
    "unit_ids": ["知识单元", "哪一份", "文档", "单元", "编号"],
    "targets": ["对象类型", "授权给", "部门", "角色", "人员", "全局", "targets"],
    "case_id": ["案件", "case"],
    "assessment_id": ["评估", "assessment", "编号"],
    "faq_id": ["FAQ", "faq", "哪一条"],
    "review_result": ["审核结果", "通过", "驳回"],
    "review_remark": ["审核意见", "备注", "理由"],
    "target_type": ["对象类型", "授权给", "部门", "角色", "人员", "全局"],
    "target_value": ["具体", "哪个部门", "哪个角色", "哪位"],
    "file_paths": ["文件", "路径", "文档"],
    "rule_code": ["规则", "规则号"],
}

REFUSAL_KEYWORDS = [
    "权限", "无权", "无法", "不能", "不可", "抱歉", "拒绝", "未获授权",
    "暂无访问", "不符合", "请联系管理", "无权访问", "不在您的",
]
ABSTAIN_KEYWORDS = [
    "暂未找到", "没有找到", "未找到明确", "知识库中没有", "无法回答", "没有相关",
    "暂未收录", "未提及", "无明确答案", "不包含相关内容", "建议联系知识管理员补充",
    # 4.1.5 三类资料不足的拒答话术
    "无法作答", "无法据此作答", "无法确定", "暂不给出结论", "口径不一致", "未覆盖所问",
    "仅部分相关", "资料不足", "缺少的信息",
]

# 出现这些词基本可以直接判定为拒答（不会误伤正常作答）
ABSTAIN_STRONG = [
    "无法作答", "无法据此作答", "暂不给出结论", "缺少的信息", "未覆盖所问",
]

# 答案结构标记：回答里带这些说明是在"作答"，不是在拒答
REPLY_STRUCTURE = ("【现象】", "【原因】", "【处理】", "【结论】", "【依据】",
                   "【功能】", "【使用方式】", "【限制】")
WRITE_TOOLS = {
    "toggle_rule", "review_case", "manage_blacklist",
    "delete_units", "set_unit_permissions", "review_faq",
    "update_unit", "import_knowledge",
}
CONFIRM_TOOLS = {"list_rules", "list_units", "get_unit", "get_assessment", "list_cases"}


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text or "")]
    return [p for p in parts if p]


def content_units(text: str) -> set[str]:
    """内容单元：中文二元组 + 英文/数字词，去掉停用词。"""
    text = CITATION_RE.sub("", text or "")
    units: set[str] = set()
    for run in CJK_RUN_RE.findall(text):
        if len(run) == 1:
            if run not in STOPWORDS:
                units.add(run)
            continue
        for i in range(len(run) - 1):
            bigram = run[i : i + 2]
            if bigram in STOPWORDS:
                continue
            units.add(bigram)
    for tok in ASCII_WORD_RE.findall(text):
        if tok.lower() not in STOPWORDS:
            units.add(tok.lower())
    return units


def coverage(sentence: str, ctx_units: set[str], min_len: int = 4) -> float:
    su = content_units(sentence)
    if not su:
        return 1.0
    hit = len(su & ctx_units)
    return hit / len(su)


def faithfulness(answer: str, contexts: list[str], threshold: float = 0.80) -> tuple[float, list[dict]]:
    ctx_units: set[str] = set()
    for c in contexts:
        ctx_units |= content_units(c)
    details = []
    sentences = [s for s in split_sentences(CITATION_RE.sub("", answer or "")) if len(norm(s)) >= 4]
    if not sentences:
        return 0.0, []
    supported = 0
    for s in sentences:
        cov = coverage(s, ctx_units)
        ok = cov >= threshold
        supported += int(ok)
        details.append({"sentence": s, "coverage": round(cov, 3), "supported": ok})
    return supported / len(sentences), details


def citation_map(reply: str) -> list[dict]:
    """返回 [{'sentence': 被引用句, 'citations': [1,2]}]"""
    out = []
    for s in split_sentences(reply or ""):
        nums = [int(n) for n in CITATION_RE.findall(s)]
        if nums:
            out.append({"sentence": CITATION_RE.sub("", s).strip(), "citations": nums})
    return out


CLAIM_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:元/天|元|天|个工作日|小时|分钟|%|次|人)")


def claim_tokens(text: str) -> list[str]:
    return [norm(m) for m in CLAIM_RE.findall(text or "")]


def claim_consistency(answer: str, contexts: list[str]) -> tuple[float, list[dict]]:
    """口径一致性：答案中出现的金额/时长/比例等关键口径是否能在片段中找到。"""
    ctx = norm(" ".join(contexts))
    claims = claim_tokens(answer)
    if not claims:
        return 1.0, []
    details = []
    hit = 0
    for c in claims:
        ok = c in ctx
        hit += int(ok)
        details.append({"claim": c, "in_context": ok})
    return hit / len(claims), details


def has_any(text: str, keywords: list[str]) -> bool:
    t = text or ""
    return any(k in t for k in keywords)


def is_refusal(text: str) -> bool:
    return has_any(text, REFUSAL_KEYWORDS)


def is_abstain(text: str) -> bool:
    """判答是否是「资料不足」拒答。

    不能只做关键词匹配：像「4.1 回答显示'暂未找到明确答案'怎么办」这种
    本身就在解释拒答机制的题目，正确答案里必然出现"暂未找到"，
    只按关键词会把它误判成拒答（进而被排除出 B2/B3/B4 统计）。

    判定顺序：
    1. 出现强拒答话术（"无法作答"/"缺少的信息"…）→ 是拒答
    2. 带答案结构标记（【现象】【结论】…）→ 是在作答，不是拒答
    3. 否则只看开头 80 字是否出现弱拒答词（拒答通常开门见山）
    """
    t = text or ""
    if has_any(t, ABSTAIN_STRONG):
        return True
    if any(m in t for m in REPLY_STRUCTURE):
        return False
    return has_any(t[:80], ABSTAIN_KEYWORDS)


def asks_for_field(text: str, field: str) -> bool:
    kws = FIELD_KEYWORDS.get(field, [field])
    return has_any(text, kws)


def field_f1(pred: dict, gold: dict) -> tuple[int, int, int]:
    """字段级：返回 (tp, n_pred, n_gold)。值比较前做规范化。"""
    pred_n = {k: norm(str(v)).lower() for k, v in (pred or {}).items() if not k.startswith("__")}
    gold_n = {k: norm(str(v)).lower() for k, v in (gold or {}).items()}
    tp = sum(1 for k, v in gold_n.items() if pred_n.get(k) == v)
    return tp, len(pred_n), len(gold_n)
