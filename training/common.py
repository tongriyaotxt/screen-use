"""torch-free 核心：轨迹加载、观测序列化、标签构建（TRAINING.md §1/§2）。

本模块不依赖 torch/transformers，可单测（tests/test_training_common.py）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ---- 动作词表（TRAINING §5.1，8 类） ----
# 单一真源在 screen_use/jev_bridge.py（运行时包不背 torch，训练侧反向复用同一份词表，
# 保证 train/serve 不漂移）；history 文本格式也由 jev_bridge.history_entry 对齐。
from screen_use.jev_bridge import OP2IDX, OPS, history_entry  # noqa: F401

# 轨迹动作 → Jev 动作；hotkey/press/think_error 不在动作集内（永升级，不进训练集）
_ACTION_MAP = {
    "click_id": "CLICK", "click": "CLICK",
    "double_click": "DOUBLE_CLICK", "right_click": "RIGHT_CLICK",
    "scroll": "SCROLL", "wait": "WAIT", "done": "DONE",
    "blocked": "BLOCKED", "type": "TYPE",
}

# ---- 锚点特殊 token（TRAINING §2） ----
SPECIAL_TOKENS = ["<|goal|>", "<|end_goal|>", "<|window|>", "<|hist|>",
                  "<|end_hist|>", "<|elem|>", "<|decide|>"]

MAX_ELEMENTS_PER_SAMPLE = 150


def load_trace_files(dirs: list[Path]) -> list[tuple[Path, list[dict]]]:
    """加载轨迹文件 → [(文件, [步, ...])]，过滤无法解析的行。"""
    out = []
    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.jsonl")):
            steps = []
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    steps.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if steps:
                out.append((f, steps))
    return out


def iter_samples(dirs: list[Path]) -> list[dict]:
    """轨迹步 → 训练样本（含标签）。跳过：非动作集动作、执行失败步、元素缺失步。

    元素槽位：元素在 elements 列表中的下标（0..n-1）；无目标元素 = n（no-element 槽）。
    span：TYPE 步文本在 goal 中的字符区间（goal-copy 机制）；找不到 → span=None（mask）。
    history：同一文件内当前步之前的动作（复建运行历史）。
    """
    samples = []
    for f, steps in load_trace_files(dirs):
        history: list[str] = []
        for s in steps:
            op = _ACTION_MAP.get(s.get("action"))
            if op is None:
                history.append(f"[决策失败 ✗]")
                continue
            elements = s.get("elements", [])[:MAX_ELEMENTS_PER_SAMPLE]
            args = s.get("args", {})

            # 目标元素槽位
            elem_slot = len(elements)  # 默认 no-element 槽
            target_id = args.get("id") or args.get("element_id")
            if target_id is None and op == "TYPE":
                target_id = _element_id_from_detail(s.get("detail", ""), elements)
            if target_id is not None:
                for i, e in enumerate(elements):
                    if e.get("id") == target_id:
                        elem_slot = i
                        break
                else:
                    # 标签指向的元素不在观测中 → 跳过（脏数据）
                    history.append(_hist_entry(op, s, ok=False))
                    continue

            # TYPE 的 goal-copy span
            span = None
            if op == "TYPE":
                text = str(args.get("text", ""))
                if text:
                    pos = s["goal"].find(text)
                    if pos >= 0:
                        span = [pos, pos + len(text)]

            samples.append({
                "goal": s["goal"],
                "foreground": s.get("foreground_title", ""),
                "elements": elements,
                "history": list(history[-4:]),  # 滑窗 4 步（TRAINING §2 截断策略）
                "op": op,
                "elem_slot": elem_slot,
                "n_elements": len(elements),
                "span": span,
                "ok": bool(s.get("ok", True)),
                "source": s.get("source", "vlm_trace"),
                "quality": s.get("quality", "silver"),
                "file": f.name,
            })
            history.append(_hist_entry(op, s, ok=bool(s.get("ok", True))))
    return samples


def _element_id_from_detail(detail: str, elements: list[dict]):
    """兼容旧 mock 数据：type 步 detail 形如 \"输入 'x' → 网址输入框\"，按名称找回元素 id。"""
    m = re.search(r"→\s*(.+)$", detail)
    if not m:
        return None
    name = m.group(1).strip()
    for e in elements:
        if e.get("name") == name:
            return e.get("id")
    return None


def _hist_entry(op: str, s: dict, ok: bool) -> str:
    """history 片段：复用运行时同一实现（jev_bridge.history_entry），train/serve 同构。"""
    return history_entry(op, s.get("args", {}), ok)


def serialize_observation(goal: str, foreground: str, elements: list[dict],
                          history: list[str]) -> str:
    """观测 → 锚点文本模板（TRAINING §2）。train/serve 共用同一函数，保证一致性。"""
    lines = [f"<|goal|>{goal}<|end_goal|>", f"<|window|>{foreground}"]
    hist = "".join(history) if history else "(无)"
    lines.append(f"<|hist|>{hist}<|end_hist|>")
    for e in elements:
        bbox = " ".join(str(int(v)) for v in e.get("bbox", [0, 0, 0, 0]))
        name = str(e.get("name", ""))[:32]  # 名称截断（TRAINING §4 压缩策略）
        lines.append(f"<|elem|>#{e['id']} {e.get('control_type', '')} \"{name}\" {bbox}")
    lines.append("<|decide|>")
    return "\n".join(lines)
