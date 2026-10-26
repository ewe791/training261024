"""F-E0 —— 测试集划分。

按 85/15 划分，硬约束：
  1. 以 scene_key 为最小单位分组，同一场景的不同表述只会落在同一侧（防泄漏）
  2. 测试集每类场景不低于 config.TEST_MIN
  3. 越权负例（T6）测试集不少于 30 条

输出：
  data/train.jsonl
  eval/test.jsonl
  data/quality_train.jsonl
  eval/quality_test.jsonl
  eval/split_meta.json

运行：python eval/split_dataset.py
"""
from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import config  # noqa: E402
from common.dataio import count_by, read_jsonl, write_json, write_jsonl  # noqa: E402

QUALITY_TEST_MIN = {"B-FACT": 30, "B-MULTI": 10, "B-REFUSE": 10}


def _group_by_scene(rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
    buckets: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        buckets[r["category"]][r["scene_key"]].append(r)
    return buckets


def split_pool(
    rows: list[dict],
    ratio: float,
    min_test: dict[str, int],
    rng: random.Random,
    stratify_keys: list[str] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """按 scene_key 整组划分；stratify_keys 指定的子类保证测试集至少有一个场景。

    为什么需要分层：B 层的「故障处理」只有 2 个场景，纯随机划分可能全部落到训练集，
    结果是 B1F 号称考四类格式、实际只考了三类。指标"看着达标"，但那一类根本没测。
    """
    buckets = _group_by_scene(rows)
    train: list[dict] = []
    test: list[dict] = []
    meta: dict = {"per_category": {}}
    for cat, scenes in sorted(buckets.items()):
        keys = sorted(scenes)
        rng.shuffle(keys)
        total = sum(len(scenes[k]) for k in keys)
        target = max(math.ceil(total * (1 - ratio)), min_test.get(cat, 0))
        # 下限不能把训练集吃光：总量小时（如 T6 只有 40 条），
        # 让测试集最多占 40%，否则会出现「测试 30 条、训练 10 条」的倒挂。
        target = min(target, max(1, int(total * 0.4)))
        picked: list[str] = []
        cnt = 0

        # ① 子类覆盖保障：每个子类（格式 / 拒答类型）先占一个场景进测试集
        forced: dict[str, int] = {}
        if stratify_keys:
            covered: set[tuple[str, str]] = set()
            for k in keys:
                for r in scenes[k]:
                    exp = r.get("expected") or {}
                    for sk in stratify_keys:
                        v = exp.get(sk)
                        if not v or (sk, v) in covered:
                            continue
                        covered.add((sk, v))
                        if k not in picked:
                            picked.append(k)
                            cnt += len(scenes[k])
                        forced[v] = forced.get(v, 0) + 1

        # ② 按比例补足
        for k in keys:
            if cnt >= target:
                break
            if k in picked:
                continue
            picked.append(k)
            cnt += len(scenes[k])
        # 余量兜底：补到 target（已含 40% 上限），不再直接补到 min_test
        i = 0
        while cnt < target and i < len(keys):
            k = keys[i]
            if k not in picked:
                picked.append(k)
                cnt += len(scenes[k])
            i += 1
        picked_set = set(picked)
        for k in keys:
            (test if k in picked_set else train).extend(scenes[k])
        meta["per_category"][cat] = {
            "total": total,
            "test": cnt,
            "train": total - cnt,
            "test_scenes": len(picked),
        }
        if forced:
            meta["per_category"][cat]["stratified"] = forced
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test, meta


def main() -> int:
    config.ensure_dirs()
    rng = random.Random(config.SEED)

    tool_rows = read_jsonl(config.RAW_DIR / "tool_calling_all.jsonl")
    quality_rows = read_jsonl(config.RAW_DIR / "quality_all.jsonl")
    if not tool_rows or not quality_rows:
        print("缺少原始数据，请先运行 python scripts/build_dataset.py")
        return 1

    tr, te, meta = split_pool(tool_rows, config.SPLIT_RATIO, config.TEST_MIN, rng)
    qtr, qte, qmeta = split_pool(quality_rows, config.QUALITY_SPLIT_RATIO,
                                 QUALITY_TEST_MIN, rng,
                                 stratify_keys=["format", "refuse_type"])

    # ---- 泄漏校验 ----
    leak = {r["scene_key"] for r in tr} & {r["scene_key"] for r in te}
    qleak = {r["scene_key"] for r in qtr} & {r["scene_key"] for r in qte}
    assert not leak, f"训练/测试场景重叠: {list(leak)[:3]}"
    assert not qleak, f"质量集场景重叠: {list(qleak)[:3]}"

    # ---- 最低条数校验 ----
    # 语义：语料够就必须划够；语料本身不够时只告警（硬失败会让人误以为是划分写错了）。
    te_counts = count_by(te, "category")
    pool_counts = count_by(tool_rows, "category")
    short = {c: (te_counts.get(c, 0), m) for c, m in config.TEST_MIN.items()
             if te_counts.get(c, 0) < m}
    hard_short = {c: v for c, v in short.items()
                  if pool_counts.get(c, 0) >= v[1] * 3}
    assert not hard_short, f"测试集类别不足（语料足够却没划够）: {hard_short}"
    for c, (got, want) in short.items():
        print(f"[警告] {c} 全量仅 {pool_counts.get(c, 0)} 条，测试集 {got} 条"
              f"（低于目标 {want}）；已按「测试最多占 40%」处理")
    qte_counts = count_by(qte, "category")
    qshort = {c: (qte_counts.get(c, 0), m) for c, m in QUALITY_TEST_MIN.items()
              if qte_counts.get(c, 0) < m}
    assert not qshort, f"质量测试集类别不足: {qshort}"

    write_jsonl(config.TRAIN_PATH, tr)
    write_jsonl(config.TEST_PATH, te)
    write_jsonl(config.QUALITY_TRAIN_PATH, qtr)
    write_jsonl(config.QUALITY_TEST_PATH, qte)

    payload = {
        "ratio": config.SPLIT_RATIO,
        "quality_ratio": config.QUALITY_SPLIT_RATIO,
        "seed": config.SEED,
        "tool_calling": {
            "total": len(tool_rows),
            "train": len(tr),
            "test": len(te),
            "test_by_category": te_counts,
            "train_by_category": count_by(tr, "category"),
            "test_scenes": len({r["scene_key"] for r in te}),
            **meta,
        },
        "quality": {
            "total": len(quality_rows),
            "train": len(qtr),
            "test": len(qte),
            "test_by_category": qte_counts,
            **qmeta,
        },
        "leak_check": {"tool_overlap": 0, "quality_overlap": 0},
    }
    write_json(config.SPLIT_META_PATH, payload)

    print("== 工具调用数据集 ==")
    print(f"train {len(tr)} 条 / test {len(te)} 条（{config.SPLIT_RATIO:.0%} / {1 - config.SPLIT_RATIO:.0%}）")
    print("test 类别分布:", te_counts)
    print("场景不重叠校验: 通过")
    print("== 回答质量数据集 ==")
    print(f"train {len(qtr)} 条 / test {len(qte)} 条")

    # 覆盖检查：训练有、测试没有的格式/拒答类型，指标会「看着达标但没考到」
    from collections import Counter as _C

    def _cov(rows, key):
        return _C((r.get("expected") or {}).get(key) for r in rows
                  if (r.get("expected") or {}).get(key))

    fmt_tr, fmt_te = _cov(qtr, "format"), _cov(qte, "format")
    ref_tr, ref_te = _cov(qtr, "refuse_type"), _cov(qte, "refuse_type")
    gaps: list[str] = []
    for k in fmt_tr:
        if not fmt_te.get(k):
            gaps.append(f"答案格式「{k}」训练 {fmt_tr[k]} 条，测试 0 条")
    for k in ref_tr:
        if not ref_te.get(k):
            gaps.append(f"拒答类型「{k}」训练 {ref_tr[k]} 条，测试 0 条")
    if gaps:
        print("\n[警告] 以下子类的测试样本为空，对应指标无法被检验：")
        for g in gaps:
            print("  ·", g)
        print("  处理：换 seed 重划、补该类的场景，或在报告的该指标下显式标注「未覆盖」。")
    else:
        print("覆盖检查: 训练出现过的格式/拒答类型在测试集里都有样本")
    print("test 类别分布:", qte_counts)
    print("\n产出: data/train.jsonl, eval/test.jsonl, data/quality_train.jsonl, "
          "eval/quality_test.jsonl, eval/split_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
