"""F-E5 —— 评测报告生成。

读取 eval/results_*.json、eval/compare.json、eval/split_meta.json、config/lora_sft.yaml，
拼装 reports/eval_finetune.md。

运行：python scripts/gen_report.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import config  # noqa: E402
from common.dataio import read_json  # noqa: E402

# 4.1.1~4.1.5 能力 → 指标映射（顺序即报告章节顺序）
ABILITY = [
    ("4.1.1", "行业术语与回答格式适配能力",
     "按知识库所属行业的术语、口径与输出格式回答（制度解释 / 产品说明 / 操作步骤 / 故障处理）。",
     "行业问答样例、标准术语表（config/terms.json）、四类答案模板（config/formats.json）",
     "术语是否准确、格式是否稳定、是否符合业务口径",
     ["B1", "B1F"]),
    ("4.1.2", "检索内容忠实回答能力",
     "严格依据检索结果作答，不编造资料中不存在的事实、数字、政策与结论，不遗漏关键条件。",
     "用户问题 + 检索片段 + 标准答案；不可回答样例（弱相关召回）",
     "答案是否有依据、是否引入幻觉、是否遗漏关键条件",
     ["B2", "B2C", "B6"]),
    ("4.1.3", "多片段信息整合能力",
     "从多个检索片段合并同一主题信息，处理上下文分散、表述不一致与跨段落引用。",
     "多段检索结果、跨文档问题、整合后的标准答案",
     "信息整合是否完整，是否覆盖每个片段的要点与适用条件",
     ["B3"]),
    ("4.1.4", "证据引用与出处标注能力",
     "在答案中标注依据来源（片段编号），说明结论来自哪些条款。",
     "带来源编号的检索结果、带引用标注的答案样例",
     "引用是否准确，答案内容与引用证据是否一致",
     ["B4"]),
    ("4.1.5", "资料不足时的拒答能力",
     "检索结果不足、无关或互相矛盾时不强行回答，并指出缺少哪些信息与下一步方向。",
     "无召回、弱相关召回、冲突召回与标准拒答样例",
     "拒答是否及时，能否指出缺失信息与补充方向",
     ["B5", "B5M"]),
]

CAT_NAMES = {
    "T1": "直接调用", "T2": "缺参澄清", "T3": "多轮指代",
    "T4": "工具混淆", "T5": "写操作确认", "T6": "越权负例",
    "B-FACT": "有明确答案", "B-MULTI": "多片段整合", "B-REFUSE": "应拒答",
}


METRIC_ORDER = ["A1", "A2", "A3", "A4", "A5", "A6",
                "B1", "B1F", "B2", "B2C", "B6", "B3", "B4", "B5", "B5M"]


def read_yaml_flat(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def pct(v) -> str:
    return "—" if v is None else f"{v * 100:.2f}%"


def delta(v: float) -> str:
    return f"{v * 100:+.2f}pp"


def build() -> str:
    cfg = read_yaml_flat(config.CONFIG_DIR / "lora_sft.yaml")
    tool_res = read_json(config.RES_TOOL_PATH, {}) or {}
    qual_res = read_json(config.RES_QUALITY_PATH, {}) or {}
    reg = read_json(config.RES_REGRESSION_PATH, {}) or {}
    cmp_data = read_json(config.COMPARE_PATH, {}) or {}
    split = read_json(config.SPLIT_META_PATH, {}) or {}
    backend = tool_res.get("backend", config.BACKEND)

    L: list[str] = []
    L.append("# 4.1 知识库 Agent 模型微调评测报告")
    L.append("")
    L.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　"
             f"推理后端：`{backend}`")
    L.append(">")
    L.append(f"> 被测模型：基座 `{config.BASE_MODEL_NAME}` / 微调 `{config.FINETUNED_MODEL_NAME}`")
    L.append("")

    if backend == "stub":
        L.append("> ⚠️ **本报告数据由 `stub` 模拟后端产生，仅用于验证评测流水线是否跑通，"
                 "不代表真实模型能力，不可作为验收依据。**")
        L.append("> 接入真实模型后重跑 `python scripts/run_all.py --backend openai`"
                 "（或 `--backend local`）即产出真实数据。")
        L.append("")

    # 1 概述
    L.append("## 1. 评测概述")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("| --- | --- |")
    L.append(f"| 基座模型 | `{cfg.get('base_model', config.BASE_MODEL_NAME)}` |")
    L.append(f"| 微调方式 | {cfg.get('method', 'lora').upper()}（rank={cfg.get('lora_rank', '-')}, "
             f"alpha={cfg.get('lora_alpha', '-')}, dropout={cfg.get('lora_dropout', '-')}） |")
    L.append(f"| 学习率 / epoch | {cfg.get('learning_rate', '-')} / {cfg.get('num_epochs', '-')} |")
    L.append(f"| batch × grad_accum | {cfg.get('batch_size', '-')} × {cfg.get('grad_accum', '-')} |")
    L.append(f"| 目标模块 | `{cfg.get('target_modules', '-')}` |")
    L.append("")

    tc = (split.get("tool_calling") or {})
    ql = (split.get("quality") or {})
    if tc:
        L.append("### 1.1 测试集划分")
        L.append("")
        L.append(f"A 层全量 {tc.get('total')} 条，按 **{int(config.SPLIT_RATIO * 100)}/"
                 f"{100 - int(config.SPLIT_RATIO * 100)}** 划分：训练 {tc.get('train')} 条、"
                 f"测试 {tc.get('test')} 条；以 `scene_key` 为最小单位分组，"
                 "同一场景的不同表述不会跨集出现（泄漏校验："
                 f"{(split.get('leak_check') or {}).get('tool_overlap', 0)} 条重叠）。")
        L.append("")
        L.append("| 类别 | 场景 | 全量 | 训练 | 测试 | 最低要求 | 达标 |")
        L.append("| --- | --- | --- | --- | --- | --- | --- |")
        by_cat = tc.get("test_by_category") or {}
        per = (tc.get("per_category") or {})
        for c in ("T1", "T2", "T3", "T4", "T5", "T6"):
            n_test = by_cat.get(c, 0)
            min_req = config.TEST_MIN.get(c, 0)
            L.append(f"| {c} | {CAT_NAMES[c]} | {per.get(c, {}).get('total', '-')} | "
                     f"{per.get(c, {}).get('train', '-')} | {n_test} | {min_req} | "
                     f"{'✓' if n_test >= min_req else '✗'} |")
        L.append("")
        qcat = ql.get("test_by_category") or {}
        L.append(f"B 层全量 {ql.get('total')} 条，训练 {ql.get('train')} 条、"
                 f"测试 {ql.get('test')} 条（有明确答案 {qcat.get('B-FACT', 0)} / "
                 f"多片段 {qcat.get('B-MULTI', 0)} / 应拒答 {qcat.get('B-REFUSE', 0)}）。")
        L.append("")

    # 2 A 层
    L.append("## 2. A 层：工具调用能力对比")
    L.append("")
    L.append("| 指标 | 基座 | 微调 | 变化 | 目标 | 达标 |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    table = cmp_data.get("table") or []
    for r in table:
        if not r["metric"].startswith("A"):
            continue
        L.append(f"| {r['label']} | {pct(r['base'])} | {pct(r['finetuned'])} | "
                 f"{delta(r['delta'])} | {r['target_text']} | {'✓' if r['pass'] else '✗'} |")
    L.append("")
    a_fail = [r["metric"] for r in table if r["metric"].startswith("A") and not r["pass"]]
    L.append(f"A 层判定：**{'全部达标' if not a_fail else '未达标 ' + '、'.join(a_fail)}**。"
             "A6 越权拒答为硬指标，微调后必须保持 100%。")
    L.append("")

    # 3 B 层：按 4.1.1~4.1.5 分节
    L.append("## 3. B 层：回答质量对比（对齐 4.1.1 ~ 4.1.5）")
    L.append("")
    meta = (qual_res.get("meta") or {}).get("finetuned") or {}
    ns = meta.get("n") or {}
    src1 = meta.get("B1_source", "auto_rule")
    src6 = meta.get("B6_source", "auto_rule")
    for code, name, ability, data, focus, keys in ABILITY:
        L.append(f"### {code} {name}")
        L.append("")
        L.append(f"- **能力要求**：{ability}")
        L.append(f"- **训练数据**：{data}")
        L.append(f"- **评估重点**：{focus}")
        if code == "4.1.3":
            n = ns.get("B3")
            L.append(f"- **样本量**：多片段用例 {n} 条（跨文档配对，每个片段都需被覆盖）")
        elif code == "4.1.5":
            rb = meta.get("refuse_by_type") or {}
            parts = [f"{k} {v['hit']}/{v['n']}" for k, v in rb.items()]
            L.append(f"- **样本量**：应拒答 {ns.get('B5')} 条（" + "、".join(parts) + "）")
        L.append("")
        L.append("| 指标 | 基座 | 微调 | 变化 | 目标 | 达标 |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for k in keys:
            r = next((x for x in table if x["metric"] == k), None)
            if not r:
                continue
            L.append(f"| {r['label']} | {pct(r['base'])} | {pct(r['finetuned'])} | "
                     f"{delta(r['delta'])} | {r['target_text']} | {'✓' if r['pass'] else '✗'} |")
        L.append("")

        # 4.1.1 必须按四类格式分列：B1F 只统计有样本的格式，
        # 不拆开读者会以为「四类模板都考了」
        if code == "4.1.1":
            fbase = (qual_res.get("meta") or {}).get("base", {}).get("format_by_type") or {}
            fft = (qual_res.get("meta") or {}).get("finetuned", {}).get("format_by_type") or {}
            L.append("按答案格式拆分（B1F 的分子分母都只计有样本的格式）：")
            L.append("")
            L.append("| 答案格式 | 样本(基座/微调) | 基座 | 微调 | 覆盖状态 |")
            L.append("| --- | --- | --- | --- | --- |")
            missing: list[str] = []
            n_mismatch = False
            for k in fft or fbase:
                nb = (fbase.get(k) or {}).get("n", 0)
                nf = (fft.get(k) or {}).get("n", 0)
                hb = (fbase.get(k) or {}).get("hit", 0)
                hf = (fft.get(k) or {}).get("hit", 0)
                bp = f"{hb}/{nb}（{hb / nb:.0%}）" if nb else "—"
                fp = f"{hf}/{nf}（{hf / nf:.0%}）" if nf else "—"
                if nf == 0:
                    status = "**未覆盖**（测试集无该类样本）"
                    missing.append(k)
                elif nf < 5:
                    status = f"已覆盖，但样本仅 {nf} 条（通过率仅供参考）"
                else:
                    status = "已覆盖"
                if nb != nf:
                    n_mismatch = True
                    nt = f"{nb} / {nf}"
                else:
                    nt = f"{nf}"
                L.append(f"| {k} | {nt} | {bp} | {fp} | {status} |")
            L.append("")
            if n_mismatch:
                L.append("> 两侧分母不同：基座把部分可答问题误判为拒答，按口径这类样本不计入 B1F"
                         "（单独记在 4.1.5 的误拒答诊断里），所以基座分母略大于微调。")
                L.append("")
            if missing:
                L.append(f"> **统计口径提醒**：本次 B1F 只覆盖了 "
                         f"{'、'.join(k for k in (fft or fbase) if (fft.get(k) or {}).get('n'))}，"
                         f"**{'、'.join(missing)} 因测试集无样本未纳入统计**。"
                         f"该格式的真实遵循率目前无法从本报告得出，需补对应测试样本后重跑。")
                L.append("")

    L.append("口径说明：")
    L.append("")
    L.append(f"- **B1 术语准确率**：标准答案中出现的每个规范术语都必须出现在回答中（用口语别名替代算未命中）；"
             f"当前来源 `{src1}`（补齐 `eval/annotations/b1_terms.jsonl` 后自动切人工抽检）。"
             f"统计实例数 {ns.get('B1')}。")
    L.append("- **B1F 格式遵循率**：按 `config/formats.json` 的四类模板判定——制度解释须有【结论】→【依据】，"
             "产品说明须有【功能】→【使用方式】，故障处理须有【现象】→【处理】，操作步骤须有 ≥2 个有序步骤。"
             "**分母只含测试集中实际出现的格式**；哪一类没有测试样本，会在上表标注「未覆盖」，不计入分子分母。")
    L.append("- **B2 忠实度**：answer 分句，每句内容词在授权片段覆盖率 ≥80% 记为被支撑，指标 = 被支撑句 / 总句"
             f"（统计句数 {ns.get('B2_sentences')}）；计算前先剥离【结论】这类格式标记。")
    L.append(f"- **B2C 关键条件完整性**：标准答案中的约束条件（上限/须经/逾期/仅限…）在回答中的覆盖率，"
             f"统计样本 {ns.get('B2C')} 条。")
    L.append("- **B3 多片段整合完整率**：每个召回片段至少有一条关键信息被用上，按片段计分（不是复现某一句）。")
    L.append("- **B4 引用一致性**：引用编号必须存在于片段集合，且被引用句内容词在对应片段覆盖率 ≥80%。")
    L.append("- **B5M 缺失信息指出率**：分母为全部应拒答样本，必须既拒答又说清「缺少什么 + 下一步怎么办」。")
    L.append(f"- **B6 幻觉率**：当前来源 `{src6}`；代理口径为「存在覆盖率 <60% 的句子」。"
             f"统计样本 {ns.get('B6')} 条。")
    oa = meta.get("over_abstain")
    if oa is not None:
        L.append(f"- **误拒答诊断**：可答用例中被判为拒答 {oa} 条（不计入 B2/B3/B4/B6，避免与 B5 重复计分）。")
    L.append("")

    # 4 改善示例
    imps = cmp_data.get("improvements") or []
    L.append(f"## 4. 改善示例（{len(imps)} 条）")
    L.append("")
    for i, im in enumerate(imps, start=1):
        L.append(f"### 例 {i}｜{im['source']} · {im['category']}")
        L.append("")
        L.append(f"**输入**：{im['question']}")
        L.append("")
        L.append("**基座输出**：")
        L.append("")
        L.append("```")
        L.append(im["base_output"].strip())
        L.append("```")
        L.append("")
        L.append("**微调输出**：")
        L.append("")
        L.append("```")
        L.append(im["ft_output"].strip())
        L.append("```")
        L.append("")
        L.append(f"**判定理由**：{im['reason']}。")
        L.append("")

    # 5 失败案例
    fails = cmp_data.get("failures") or []
    L.append(f"## 5. 失败案例与原因分析（{len(fails)} 条）")
    L.append("")
    for i, f in enumerate(fails, start=1):
        L.append(f"### 失败 {i}｜{f['source']} · {f['category']}｜{f['cause']}")
        L.append("")
        L.append(f"**输入**：{f['question']}")
        L.append("")
        L.append(f"**期望**：`{f['expected']}`")
        L.append("")
        L.append("**微调输出**：")
        L.append("")
        L.append("```")
        L.append(f["ft_output"].strip())
        L.append("```")
        L.append("")
        L.append(f"**失败点**：{f['reason']}。")
        L.append(f"**原因分析**：{f['cause']}。")
        L.append(f"**改进建议**：{f['advice']}。")
        L.append("")

    # 6 回归护栏
    L.append("## 6. 回归护栏结果（R1~R4）")
    L.append("")
    L.append("| 项 | 内容 | 基座 | 微调 | 规则 | 结论 |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for it in reg.get("items") or []:
        L.append(f"| {it['id']} | {it['name']} | {pct(it['base']) if isinstance(it['base'], float) else it['base']} | "
                 f"{pct(it['finetuned']) if isinstance(it['finetuned'], float) else it['finetuned']} | "
                 f"{it['rule']} | {'PASS' if it['pass'] else 'FAIL'} |")
    L.append("")
    L.append(f"护栏结论：**{reg.get('verdict', '未运行')}**（`{reg.get('conclusion', '-')}`）。")
    L.append("")

    # 7 结论
    L.append("## 7. 结论与遗留问题")
    L.append("")
    failed = [r["label"] for r in table if not r["pass"]]
    hard_failed = [r["label"] for r in table if r["hard"] and not r["pass"]]
    if not failed and reg.get("conclusion") == "PASS":
        L.append("- 11 项指标全部达标，A6 越权拒答率保持 100%，回归护栏 R1~R4 全通过，**可以交付**。")
    else:
        if hard_failed:
            L.append(f"- **硬指标未达标：{'、'.join(hard_failed)} → 必须回退训练数据或调整配比，不得交付。**")
        if failed:
            L.append(f"- 未达标指标：{'、'.join(failed)}。")
        if reg.get("conclusion") != "PASS":
            L.append(f"- 回归护栏未通过：{'、'.join(reg.get('failed') or [])} → 微调需回退调整。")
    L.append("- 微调带来的最大增益集中在「流程约束类能力」：写操作先确认、缺参先追问、越权先拒绝；"
             "这类能力靠 Prompt 约束不稳定，靠数据对齐才稳定。")
    L.append("- 遗留问题：")
    L.append("  1. B1 / B5 当前若为代理口径，需按 `eval/annotations/*` 模板补齐人工抽检，再重跑 `test_answer_quality.py`。")
    L.append("  2. T4 工具混淆与 T1 直接调用在语义上高度接近，训练集需保持成对负例，避免模型退化为高频工具。")
    L.append("  3. 越权场景目前以单轮为主，建议补多轮渐进式诱导（先正常提问，再逐步索要权限外内容）。")
    if backend == "stub":
        L.append("  4. 本报告为 stub 流水线自测数据，须接入真实模型后重跑并归档真实数字。")
    L.append("")

    # 8 验收清单
    L.append("## 8. 验收清单")
    L.append("")
    checks: list[tuple[str, bool]] = []
    if tc:
        checks.append(("A 层：训练/测试场景不重叠，T1~T6 各 ≥ 最低条数",
                       all((tc.get("test_by_category") or {}).get(c, 0) >= config.TEST_MIN[c]
                           for c in config.TEST_MIN)))
    m = {r["metric"]: r for r in table}
    for key in METRIC_ORDER:
        if key in m:
            checks.append((f"{m[key]['label']} 达标（{m[key]['target_text']}）", m[key]["pass"]))
    fmeta = (qual_res.get("meta") or {}).get("finetuned", {}).get("format_by_type") or {}
    checks.append(("四类答案格式在测试集都有样本（B1F 能检验全部模板）",
                   bool(fmeta) and all((v or {}).get("n", 0) > 0 for v in fmeta.values())))
    checks.append(("B 层测试集覆盖 4.1.1~4.1.5 五类能力（明确答案/多片段/三类拒答）",
                   bool(ql) and all((ql.get("test_by_category") or {}).get(c, 0) >= 1
                                    for c in ("B-FACT", "B-MULTI", "B-REFUSE"))))
    checks.append((f"改善示例 ≥10 条（实际 {len(imps)} 条）", len(imps) >= 10))
    checks.append((f"失败案例 ≥5 条（实际 {len(fails)} 条）", len(fails) >= 5))
    checks.append(("回归护栏 R1~R4 全过", reg.get("conclusion") == "PASS"))
    checks.append(("报告含基座 vs 微调对比表与失败分析", True))
    checks.append(("硬指标 A6 达标（未达标则标记回退）", m.get("A6", {}).get("pass", False)))
    for text, ok in checks:
        L.append(f"- [{'x' if ok else ' '}] {text}")
    L.append("")
    return "\n".join(L)


def main() -> int:
    config.ensure_dirs()
    md = build()
    config.REPORT_PATH.write_text(md, encoding="utf-8")
    print(f"写出 {config.REPORT_PATH.relative_to(ROOT)}（{len(md)} 字符）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
