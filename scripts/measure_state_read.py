"""状态位读取代价实测（只读，不动鼠标/不改系统）。

决定 ④b 里到底读几个状态位：UIA 每次跨进程属性读取都是一趟 COM 往返，
逐节点读会变成采集器的延迟瓶颈（BENCH §6 已证：观测慢的根因就是每节点成本）。

用法（项目根）：
    ./.venv/Scripts/python.exe scripts/measure_state_read.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from screen_use.tools import ScreenUse  # noqa: E402


def _read(ctrl, kind: str):
    """读一个状态；不支持就返回 None（异常也要计入成本——它同样是 COM 往返）。"""
    try:
        if kind == "enabled":
            return bool(ctrl.IsEnabled)
        if kind == "checked":
            return int(ctrl.GetTogglePattern().ToggleState)
        if kind == "expanded":
            return int(ctrl.GetExpandCollapsePattern().ExpandCollapseState)
        if kind == "selected":
            return bool(ctrl.GetSelectionItemPattern().IsSelected)
    except Exception:
        return None
    return None


def timed(els, kinds: list[str]) -> tuple[float, dict]:
    """串行读 kinds 里的状态，返回 (总毫秒, 各状态的命中数)。"""
    hits = {k: 0 for k in kinds}
    t0 = time.perf_counter()
    for el in els:
        for k in kinds:
            if _read(el.control, k) is not None:
                hits[k] += 1
    return (time.perf_counter() - t0) * 1000, hits


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    su = ScreenUse()
    t0 = time.perf_counter()
    els = su.list_ui_elements_raw(max_elements=600)
    enum_ms = (time.perf_counter() - t0) * 1000
    print(f"枚举到 {len(els)} 个元素（含缓存，{enum_ms:.0f} ms）")

    # 预热一次（首次 COM 调用含连接开销，不该算进单元素成本）
    timed(els[:5], ["enabled", "checked"])

    print(f"\n{'状态位':10s} {'总 ms':>8s} {'每元素 ms':>10s} {'支持数':>7s}")
    single = {}
    for kind in ("enabled", "checked", "expanded", "selected"):
        ms, hits = timed(els, [kind])
        single[kind] = ms
        print(f"{kind:10s} {ms:8.1f} {ms/max(len(els),1):10.3f} {hits[kind]:7d}")

    allk = ["enabled", "checked", "expanded", "selected"]
    ms, hits = timed(els, allk)
    print(f"{'四位一起':10s} {ms:8.1f} {ms/max(len(els),1):10.3f}")
    ms2, _ = timed(els, ["enabled", "checked"])
    print(f"{'两位(常用)':10s} {ms2:8.1f} {ms2/max(len(els),1):10.3f}")

    print(f"\n对照：整轮观测枚举 {enum_ms:.0f} ms")
    print(f"加四位状态 → 观测变慢 {ms/enum_ms:.0%}；加两位 → {ms2/enum_ms:.0%}")
    print("\n判据：新增开销 >30% 就只读 enabled+checked（对决策影响最大的两个）")


if __name__ == "__main__":
    main()
