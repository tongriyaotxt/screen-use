"""观测成本体检：UIA 枚举的钱花在哪个窗口（纯只读，不动鼠标/不抢焦点/不外发）。

接入 Jev 后决策只要 ~0.04s/步，观测（UIA 枚举）却要 ~0.7s/步 —— 新瓶颈在这里。
优化前先看清拆分：哪个窗口吃了多少节点、多少毫秒。

用法（项目根，.venv）：
    ./.venv/Scripts/python.exe scripts/measure_observe.py --samples 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from screen_use.perception import screen, uia_tree  # noqa: E402
from screen_use.tools import ScreenUse  # noqa: E402


def measure_enum(samples: int) -> list[dict]:
    runs = []
    for _ in range(samples):
        el = uia_tree.list_ui_elements()
        runs.append({"timings": el.timings, "n_elements": len(el)})
    return runs


def measure_shot(tools: ScreenUse, samples: int) -> dict:
    """对比三种观测：仅枚举 / 截图 / 截图+SoM+JPEG 编码。"""
    enum_ms, shot_ms, full_ms = [], [], []
    for _ in range(samples):
        t0 = time.perf_counter()
        tools.list_ui_elements()
        enum_ms.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        tools.screenshot(annotate=False, max_size=tools.settings.vlm_max_size)
        shot_ms.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        tools.screenshot(annotate=True, max_size=tools.settings.vlm_max_size)
        full_ms.append((time.perf_counter() - t0) * 1000)
    return {"enum": statistics.median(enum_ms), "shot": statistics.median(shot_ms),
            "shot_som_encode": statistics.median(full_ms)}


def measure_budgets(samples: int, budgets: list[int]) -> list[dict]:
    """不同时间预算下的「速度 vs 元素表完整度」权衡（选默认值靠这张表）。"""
    rows = []
    for budget in budgets:
        total, nodes, elems, skipped, prints = [], [], [], [], []
        for _ in range(samples):
            el = uia_tree.list_ui_elements(budget_ms=budget)
            total.append(el.timings["total_ms"])
            nodes.append(el.timings["nodes"])
            elems.append(len(el))
            skipped.append(len(el.timings.get("skipped", [])))
            prints.append({(e.name, e.control_type, tuple(e.bbox)) for e in el})
        rows.append({
            "budget_ms": budget,
            "total_ms": round(statistics.median(total), 1),
            "nodes": round(statistics.median(nodes)),
            "elements": round(statistics.median(elems)),
            "skipped_windows": round(statistics.median(skipped)),
            "sets": prints,
        })
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--budgets", default="0,150,250,400,800",
                   help="对比这些预算（毫秒，0=不限）")
    args = p.parse_args()

    runs = measure_enum(args.samples)
    agg: dict[str, dict] = defaultdict(lambda: {"nodes": 0, "elements": 0, "ms": 0.0,
                                                "hits": 0, "fg": False})
    for r in runs:
        for w in r["timings"].get("windows", []):
            key = w["title"] or "(无标题)"
            a = agg[key]
            a["nodes"] += w["nodes"]
            a["elements"] += w["elements"]
            a["ms"] += w["ms"]
            a["hits"] += 1
            a["fg"] = a["fg"] or w["foreground"]

    n = max(len(runs), 1)
    print(f"UIA 枚举 {n} 次（每次平均）")
    print(f"  总耗时    {statistics.mean([r['timings']['total_ms'] for r in runs]):7.1f} ms"
          f"   (中位 {statistics.median([r['timings']['total_ms'] for r in runs]):.1f})")
    print(f"  顶层窗口收集 {statistics.mean([r['timings']['windows_ms'] for r in runs]):7.1f} ms"
          f"   (GetChildren + IsOffscreen + BoundingRectangle)")
    print(f"  访问节点  {statistics.mean([r['timings']['nodes'] for r in runs]):7.0f} 个"
          f"   (预算 {uia_tree.MAX_VISITED})")
    print(f"  产出元素  {statistics.mean([r['n_elements'] for r in runs]):7.1f} 个"
          f"   (上限 {uia_tree.MAX_ELEMENTS}）")
    print()
    print(f"{'窗口':<38}{'节点':>7}{'元素':>7}{'ms/次':>9}  前台")
    for title, a in sorted(agg.items(), key=lambda kv: -kv[1]["ms"]):
        print(f"{title[:36]:<38}{a['nodes'] / a['hits']:>7.0f}{a['elements'] / a['hits']:>7.1f}"
              f"{a['ms'] / a['hits']:>9.1f}  {'✓' if a['fg'] else ''}")

    tools = ScreenUse(dry_run=True)
    shots = measure_shot(tools, args.samples)
    print()
    print("三种观测（中位，本机）")
    print(f"  仅 UIA 枚举（Jev 快路径）      {shots['enum']:7.1f} ms")
    print(f"  截图（无标注）                 {shots['shot']:7.1f} ms")
    print(f"  截图 + SoM 标注 + JPEG 编码    {shots['shot_som_encode']:7.1f} ms"
          f"   ← 纯 VLM 路径每步都要")

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    rows = measure_budgets(args.samples, budgets)
    print()
    print("时间预算的权衡（前台窗口永远不限时；后台窗口按历史成本从便宜到贵走）")
    print(f"{'预算(ms)':>9}{'枚举(ms)':>10}{'节点':>7}{'元素':>7}{'跳过窗口':>9}")
    for r in rows:
        tag = "（不限）" if r["budget_ms"] == 0 else ""
        print(f"{r['budget_ms']:>9}{r['total_ms']:>10.0f}{r['nodes']:>7}{r['elements']:>7}"
              f"{r['skipped_windows']:>9}  {tag}")
    base = next((r for r in rows if r["budget_ms"] == 0), None)
    if base:
        print()
        for r in rows:
            speed = f"{base['total_ms'] / r['total_ms']:.1f}×" if r["total_ms"] else "-"
            print(f"  预算{r['budget_ms']:>4}  {speed:>6}  元素 {r['elements']}")
        print()
        print("注：桌面在变（窗口开关/内容刷新），同一档的元素数本身有 ±20% 波动——")
        print("    这张表看**量级**：预算把观测从秒级压到百毫秒级，代价是后台窗口的完整度。")
    print()
    print(f"当前配置 uia_budget_ms={tools.settings.uia_budget_ms}"
          f"（.env：SCREEN_USE_UIA_BUDGET_MS；0=不限）")


if __name__ == "__main__":
    main()
