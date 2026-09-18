"""desktop-jev 阶段 1：轨迹数据基线分析。

统计 ~/.screen_use/traces_full/ 下所有轨迹：
- 动作分布（哪种动作占多少步）
- **小模型覆盖率假设验证**：无文本步占比（desktop-jev 动作集可覆盖的步数比例）
  —— 这是 DESIGN §1 的 kill criterion：占比太低则 70-90% VLM 削减不成立

用法：./.venv/Scripts/python.exe scripts/analyze_traces.py [--data-quality]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

TRACES_DIR = Path.home() / ".screen_use" / "traces_full"

# desktop-jev 收窄动作集（DESIGN §5.1）：小模型可独立决策的步
COVERABLE = {"click_id", "click", "double_click", "right_click", "scroll", "wait", "done"}
# 受限可覆盖：文本从 goal copy 时才行（此处统计全部 type，需人工估计可 copy 比例）
RESTRICTED = {"type"}
# 永升级：hotkey/press（低频语义模糊）+ think_error（决策失败）
ESCALATE = {"hotkey", "press", "think_error"}


def load_steps():
    for f in sorted(TRACES_DIR.glob("*.jsonl")):
        steps = []
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        yield f, steps


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    files = list(load_steps())
    total = sum(len(s) for _, s in files)
    print(f"轨迹文件: {len(files)} 个, 总步数: {total}\n")

    # 逐任务概览
    print(f"{'文件':<42} {'步数':>4} {'OK率':>6}  目标")
    for f, steps in files:
        ok_rate = sum(1 for s in steps if s["ok"]) / max(len(steps), 1)
        goal = (steps[0]["goal"] if steps else "")[:36]
        print(f"{f.name:<42} {len(steps):>4} {ok_rate:>6.0%}  {goal}")

    # 动作分布 + 覆盖率（排除 think_error 噪声）
    dist: Counter = Counter()
    for _, steps in files:
        for s in steps:
            dist[s["action"]] += 1
    valid = total - dist.get("think_error", 0)
    coverable = sum(v for k, v in dist.items() if k in COVERABLE)
    restricted = sum(v for k, v in dist.items() if k in RESTRICTED)
    escalate = sum(v for k, v in dist.items() if k in ESCALATE - {"think_error"})

    print(f"\n动作分布（总 {total} 步）:")
    for action, n in dist.most_common():
        tag = ("小模型可覆盖" if action in COVERABLE else
               "受限可覆盖" if action in RESTRICTED else
               "永升级" if action in ESCALATE else "?")
        print(f"  {action:<12} {n:>4} 步  {n / total:>6.1%}  {tag}")

    print(f"\n=== kill criterion：小模型覆盖率（有效步 {valid}）===")
    print(f"  直接可覆盖（点击/滚动/等待/完成）: {coverable:>4}  {coverable / valid:>6.1%}")
    print(f"  受限可覆盖（type，文本需可从 goal copy）: {restricted:>4}  {restricted / valid:>6.1%}")
    print(f"  永升级（hotkey/press）: {escalate:>4}  {escalate / valid:>6.1%}")
    print(f"  → 乐观覆盖率（含受限 type）: {(coverable + restricted) / valid:.1%}")
    print(f"  → 保守覆盖率（type 全升级）: {coverable / valid:.1%}")
    print("\n注: 当前样本量小且以计算器/记事本为主，结论仅供方向性参考；"
          "覆盖率随任务类型变化，需持续采集。")


if __name__ == "__main__":
    main()
