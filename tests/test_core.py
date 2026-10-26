"""核心逻辑单测：协议解析、字段 F1、忠实度、拒答判定、划分防泄漏。

运行：python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.formatting import (  # noqa: E402
    build_formatted,
    check_format,
    condition_coverage,
    detect_conflict,
    fragment_coverage,
    strip_markers,
    table_points,
    term_check,
)
from common.protocol import parse_output  # noqa: E402
from common.text import (  # noqa: E402
    asks_for_field,
    claim_consistency,
    field_f1,
    faithfulness,
    is_abstain,
    is_refusal,
)


class TestProtocol(unittest.TestCase):
    def test_tool_call(self):
        p = parse_output('{"type":"tool_call","name":"risk_check","arguments":{"user_id":"U1001"}}')
        self.assertEqual(p.kind, "tool_call")
        self.assertEqual(p.name, "risk_check")
        self.assertEqual(p.arguments["user_id"], "U1001")
        self.assertTrue(p.json_valid)

    def test_clarify(self):
        p = parse_output('{"type":"clarify","missing":["order_id"],"reply":"请提供订单号"}')
        self.assertEqual(p.kind, "clarify")
        self.assertIn("order_id", p.missing)

    def test_refuse(self):
        p = parse_output('{"type":"refuse","reason":"permission","reply":"缺少权限，无法执行"}')
        self.assertEqual(p.kind, "refuse")
        self.assertTrue(is_refusal(p.reply))

    def test_fenced_json(self):
        p = parse_output('```json\n{"type":"answer","reply":"上限 600 元。[1]"}\n```')
        self.assertEqual(p.kind, "answer")
        self.assertEqual(p.citations, [1])

    def test_json_with_noise(self):
        p = parse_output('好的，结果如下 {"type":"tool_call","name":"list_rules","arguments":{}} 以上')
        self.assertEqual(p.kind, "tool_call")
        self.assertEqual(p.name, "list_rules")

    def test_malformed(self):
        p = parse_output('{"type":"tool_call","name":"risk_check","arguments":{"user_id":')
        self.assertFalse(p.json_valid)
        self.assertFalse(p.has_tool_call)

    def test_native_tool_calls(self):
        native = [{"function": {"name": "get_unit", "arguments": '{"unit_id":"KU-1"}'}}]
        p = parse_output("", native)
        self.assertEqual(p.name, "get_unit")
        self.assertEqual(p.arguments["unit_id"], "KU-1")


class TestMetrics(unittest.TestCase):
    def test_field_f1(self):
        tp, npred, ngold = field_f1(
            {"event_type": "order_create", "user_id": "U1001", "order_id": "O2009"},
            {"event_type": "order_create", "user_id": "U1001", "order_id": "O2001"},
        )
        self.assertEqual((tp, npred, ngold), (2, 3, 3))

    def test_field_f1_empty_pred(self):
        tp, npred, ngold = field_f1({}, {"a": "1", "b": "2"})
        self.assertEqual((tp, npred, ngold), (0, 0, 2))

    def test_faithfulness_supported(self):
        ctx = ["一线城市每晚上限 600 元，二线城市 450 元，其他城市 350 元。"]
        score, details = faithfulness("一线城市每晚上限 600 元。[1]", ctx)
        self.assertEqual(score, 1.0)
        self.assertTrue(details[0]["supported"])

    def test_faithfulness_hallucination(self):
        ctx = ["一线城市每晚上限 600 元。"]
        score, _ = faithfulness("一线城市每晚上限 600 元；超标部分按 1200 元报销，无需审批。", ctx)
        self.assertLess(score, 1.0)

    def test_claim_consistency(self):
        ctx = ["一线城市 600 元，二线城市 450 元。"]
        ok, _ = claim_consistency("一线城市 600 元，二线 450 元。[1]", ctx)
        self.assertEqual(ok, 1.0)
        bad, details = claim_consistency("一线城市 800 元。[1]", ctx)
        self.assertEqual(bad, 0.0)
        self.assertFalse(details[0]["in_context"])

    def test_abstain_vs_refusal(self):
        self.assertTrue(is_abstain("知识库中暂未找到明确答案，建议联系知识管理员补充相关制度。"))
        # 正常回答里出现"权限"不应被判为拒答
        self.assertFalse(is_abstain("知识单元默认不对外可见，需要管理员显式配置权限。[1]"))
        self.assertTrue(is_refusal("该操作超出您的权限范围，我无法执行。"))

    def test_asks_for_field(self):
        self.assertTrue(asks_for_field("还需要您提供订单号", "order_id"))
        self.assertFalse(asks_for_field("已为您停止规则", "order_id"))


class TestFormatting(unittest.TestCase):
    """4.1.1~4.1.5 的确定性判定。"""

    def test_format_pass(self):
        reply = "【结论】一线城市 600 元。\n【依据】住宿费凭发票实报实销。[1]"
        self.assertTrue(check_format(reply, "制度解释")["pass"])

    def test_format_missing_marker(self):
        ok = check_format("一线城市 600 元，凭发票报销。[1]", "制度解释")
        self.assertFalse(ok["pass"])
        self.assertIn("【结论】", ok["missing"])

    def test_format_out_of_order(self):
        self.assertFalse(check_format("【依据】x\n【结论】y", "制度解释")["pass"])

    def test_format_steps(self):
        self.assertTrue(check_format("1. 填写报销单\n2. 上传发票\n3. 部门审批[1]", "操作步骤")["pass"])
        self.assertFalse(check_format("先填写报销单再上传发票", "操作步骤")["pass"])

    def test_strip_markers(self):
        self.assertEqual(strip_markers("【结论】上限 600 元"), "上限 600 元")
        self.assertEqual(strip_markers("1. 填写报销单"), "填写报销单")

    def test_condition_coverage(self):
        cond = "超过 3 小时或高铁不便的可乘坐飞机经济舱，需提前 3 个工作日报部门负责人审批"
        full, _ = condition_coverage("超过 3 小时可乘坐飞机经济舱，需提前 3 个工作日报部门负责人审批", [cond])
        self.assertEqual(full, 1.0)
        miss, _ = condition_coverage("一线城市每晚上限 600 元", [cond])
        self.assertLess(miss, 0.6)

    def test_fragment_coverage(self):
        groups = [["一线城市每晚上限 600 元"], ["工作时间为周一至周五 9:00-18:00"]]
        ok, _ = fragment_coverage(
            "一线城市每晚上限 600 元；工作时间为周一至周五 9:00-18:00", groups)
        self.assertEqual(ok, 1.0)
        half, _ = fragment_coverage("一线城市每晚上限 600 元", groups)
        self.assertEqual(half, 0.5)

    def test_detect_conflict(self):
        a = "差旅住宿标准：一线城市每晚上限 600 元，二线城市 450 元。"
        b = "差旅住宿标准：一线城市每晚上限 1200 元，二线城市 900 元。"
        self.assertTrue(detect_conflict([a, b])["conflict"])
        self.assertFalse(detect_conflict([a, "工作时间为周一至周五 9:00-18:00。"])["conflict"])

    def test_term_check_alias(self):
        ctx = ["差旅住宿标准按城市分三档，一线城市每晚上限 600 元。"]
        bad = term_check("住宿费标准是一线 600 元。", ["差旅住宿标准"], ctx)
        self.assertFalse(bad["pass"])
        self.assertTrue(bad["alias_used"])
        good = term_check("差旅住宿标准：一线城市每晚上限 600 元。", ["差旅住宿标准"], ctx)
        self.assertTrue(good["pass"])

    def test_table_points_natural_language(self):
        """表格必须转成人话，不能把 | 行当句子（否则模型学输出表格残片）。"""
        t = ("| 城市档次 | 覆盖城市 | 每晚上限（元） |\n"
             "| --- | --- | --- |\n"
             "| 一线城市 | 北京、上海 | 600 |\n")
        pts = table_points(t)
        self.assertEqual(len(pts), 1)
        self.assertFalse("|" in pts[0])
        self.assertIn("600", pts[0])
        self.assertIn("元", pts[0])
        self.assertIn("一线城市", pts[0])

    def test_fault_no_fabricated_cause(self):
        """FAQ 正文只有"处理办法"时，不能硬贴【原因】标签（会教模型编造因果）。"""
        reply = build_formatted("故障处理", ["推荐使用 Chrome 浏览器", "清除缓存后重试"],
                                [1], topic="门户登录显示异常")
        self.assertNotIn("【原因】", reply)
        self.assertIn("【现象】门户登录显示异常", reply)
        self.assertIn("【处理】推荐使用 Chrome 浏览器", reply)
        self.assertTrue(check_format(reply, "故障处理")["pass"])

    def test_fault_with_real_cause(self):
        pts = ["主要原因是项目预算年底统筹收回", "联系预算科重新下达"]
        reply = build_formatted("故障处理", pts, [1], topic="预算项目查询不到")
        self.assertIn("【原因】主要原因是项目预算年底统筹收回", reply)
        self.assertIn("【处理】联系预算科重新下达", reply)

    def test_chapter_no_strip(self):
        """章号必须连顿号一起剥掉，否则会问出「、门户登录…」这种残句。"""
        from scripts.build_dataset import CHAPTER_NO_RE

        self.assertEqual(CHAPTER_NO_RE.sub("", "1、门户登录显示异常").strip(), "门户登录显示异常")
        self.assertEqual(CHAPTER_NO_RE.sub("", "2.1 登录").strip(), "登录")

    def test_build_formatted_roundtrip(self):
        reply = build_formatted("制度解释", ["一线城市每晚上限 600 元", "凭发票实报实销"], [1])
        self.assertTrue(check_format(reply, "制度解释")["pass"])
        self.assertIn("[1]", reply)


class TestCloudPaths(unittest.TestCase):
    """上云时最容易踩的路径解析：配置里是 HF id，.env 里是本地目录。"""

    def test_base_model_priority(self):
        import os

        from train.finetune_lora import resolve_base_model

        cfg = {"base_model": "Qwen/Qwen2.5-7B-Instruct"}
        old = os.environ.pop("FTKB_BASE_MODEL_PATH", None)
        try:
            self.assertEqual(resolve_base_model(None, cfg), "Qwen/Qwen2.5-7B-Instruct")
            os.environ["FTKB_BASE_MODEL_PATH"] = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
            self.assertEqual(resolve_base_model(None, cfg),
                             "/root/autodl-tmp/Qwen2.5-7B-Instruct")
            self.assertEqual(resolve_base_model("/data/m", cfg), "/data/m")
        finally:
            os.environ.pop("FTKB_BASE_MODEL_PATH", None)
            if old is not None:
                os.environ["FTKB_BASE_MODEL_PATH"] = old

    def test_empty_env_falls_back_to_config(self):
        import os

        from train.finetune_lora import resolve_base_model

        old = os.environ.pop("FTKB_BASE_MODEL_PATH", None)
        try:
            os.environ["FTKB_BASE_MODEL_PATH"] = "   "   # 只有空格视为未设置
            self.assertEqual(resolve_base_model(None, {"base_model": "x/y"}), "x/y")
        finally:
            os.environ.pop("FTKB_BASE_MODEL_PATH", None)
            if old is not None:
                os.environ["FTKB_BASE_MODEL_PATH"] = old


class TestSplit(unittest.TestCase):
    def test_no_scene_leak(self):
        import random

        from eval.split_dataset import split_pool

        rows = []
        for i in range(60):
            scene = f"S{i // 3}"          # 每场景 3 条不同表述
            rows.append({"id": f"x{i}", "category": "T1", "scene_key": scene})
        train, test, meta = split_pool(rows, 0.85, {"T1": 10}, random.Random(7))
        train_scenes = {r["scene_key"] for r in train}
        test_scenes = {r["scene_key"] for r in test}
        self.assertFalse(train_scenes & test_scenes)
        self.assertGreaterEqual(len(test), 10)
        self.assertEqual(len(train) + len(test), len(rows))


if __name__ == "__main__":
    unittest.main()
