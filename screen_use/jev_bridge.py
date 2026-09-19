"""Jev ↔ VisualLoop 桥接层（cascade protocol 中环接入点，阶段 C）。

职责边界：本模块**不含 torch**，只做纯数据翻译，因此可以被
`screen_use`（运行时包）安全 import；真正的推理引擎在 `training/serve.py`，
由 `load_engine()` 懒加载（未装 torch / 加载失败 → None，自动退化成纯 VLM 循环）。

三件事：
1. 动作词表单一真源：`OPS` 由本模块定义，`training/common.py` 反向复用，
   保证 train/serve 不会各自漂移（TRAINING §2「train/serve 一致性」）。
2. `to_action()`：JevDecision（8 类 op + 元素 id + 文本）→ VisualLoop 动作 dict。
   不可执行的决策返回 None = 升级慢层（VLM），绝不猜。
3. `history_entry()`：运行时一步决策 → 训练同款 history 片段
   （与 `training.common._hist_entry` 逐字符同构，单测锁死）。

Jev v0 的诚实边界（见 research/desktop-jev/DESIGN.md §5.1）：
- SCROLL / WAIT 没有载荷（格数、秒数）输出头 → 用默认值，属已知精度损失；
- 自由文本（TYPE 的内容）只能从 goal 里 copy，copy 不到就升级，不编文本；
- BLOCKED = 认输，直接交宿主，不再调用 VLM（省一轮最贵的调用）。
"""

from __future__ import annotations

# ---- 动作词表（canonical：training.common.OPS 从这里 import） ----
OPS = ["CLICK", "DOUBLE_CLICK", "RIGHT_CLICK", "SCROLL", "WAIT", "DONE", "BLOCKED", "TYPE"]
OP2IDX = {op: i for i, op in enumerate(OPS)}

# Jev op → VisualLoop action；None = 无对应可执行动作（升级）
JEV_TO_ACTION: dict[str, str | None] = {
    "CLICK": "click_id",
    "DOUBLE_CLICK": "double_click",
    "RIGHT_CLICK": "right_click",
    "SCROLL": "scroll",
    "WAIT": "wait",
    "DONE": "done",
    "TYPE": "type",
    "BLOCKED": None,
}

# VisualLoop action → Jev op（history 文本用；hotkey/press/think_error 不在 Jev 动作集内）
ACTION_TO_JEV: dict[str, str] = {
    "click_id": "CLICK",
    "click": "CLICK",
    "double_click": "DOUBLE_CLICK",
    "right_click": "RIGHT_CLICK",
    "scroll": "SCROLL",
    "wait": "WAIT",
    "done": "DONE",
    "blocked": "BLOCKED",
    "type": "TYPE",
}

# 无载荷动作的默认值（v0 动作集没有格数/秒数头）
DEFAULT_SCROLL_CLICKS = -3
DEFAULT_WAIT_SECONDS = 1.0

# 需要元素 id 才能执行的动作
_ELEMENT_ACTIONS = ("click_id", "double_click", "right_click")


def ood_check(n_elements: int, calib: dict | None) -> str | None:
    """观测是否超出训练分布（超出 → 返回原因文本，调用方直接升级慢层）。

    2026-09-19 real-domain 实测：mock 训练集元素数中位 14 / p95 ~25，而真实桌面
    中位 84（最大 107），token 中位 2380 vs 525 —— 在这类观测上 Jev 的 joint_acc
    从 0.79 掉到 0.15、升级率 90%。既然它必然升级，就别花那 0.45s 去问它；
    "只在自己见过的尺度上做决策" 也是置信度诚实性的一部分。

    calib 里没有训练分布统计（旧 ckpt）时不做门控。
    阈值取训练集的**最大值**（不是 p95）：只有训练时真没见过的观测尺度才拦，
    分布内的长尾仍交给置信度头判断，不重复收税。
    """
    if not calib:
        return None
    stats = calib.get("train_stats") or {}
    limit = stats.get("elements_max")
    if limit and n_elements > limit:
        return (f"观测超出训练分布：元素数 {n_elements} > 训练集最大值 {int(limit)}"
                f"（真实桌面常见 80+，mock 训练集中位 {int(stats.get('elements_p50', 0))}）")
    return None


def to_action(decision: dict) -> dict | None:
    """JevDecision → VisualLoop 动作 dict。

    返回 None 表示这条决策**不可执行**（缺元素/缺文本/认输），调用方应升级慢层。
    """
    op = str(decision.get("action", "")).upper()
    if op not in JEV_TO_ACTION:
        return None
    action = JEV_TO_ACTION[op]
    if action is None:
        return None  # BLOCKED

    conf = float(decision.get("confidence") or 0.0)
    out: dict = {"action": action, "thought": f"jev:{op} conf={conf:.2f}"}

    if action in _ELEMENT_ACTIONS:
        element_id = decision.get("element_id")
        if element_id is None:
            return None  # 需要元素却没选中 → 不猜，升级
        out["id"] = int(element_id)
    elif action == "type":
        text = decision.get("text")
        if not text:
            return None  # TRAINING §1：宁可认输，不教它编文本
        out["text"] = str(text)
        element_id = decision.get("element_id")
        if element_id is not None:
            out["id"] = int(element_id)
    elif action == "scroll":
        out["clicks"] = int(decision.get("clicks") or DEFAULT_SCROLL_CLICKS)
    elif action == "wait":
        out["seconds"] = float(decision.get("seconds") or DEFAULT_WAIT_SECONDS)
    elif action == "done":
        out["result"] = str(decision.get("result") or "Jev 判定任务完成")
    return out


def history_entry(action: str, args: dict, ok: bool) -> str:
    """运行时一步 → 训练同款 history 片段（`training.common._hist_entry` 的孪生实现）。

    训练侧约定（training/common.iter_samples）：动作集外的步（hotkey/press/think_error）
    在 history 里记作「决策失败 ✗」，而不是按原名记录——serve 侧必须照抄，
    否则 history 分布和训练时对不上。
    """
    op = ACTION_TO_JEV.get(action)
    if op is None:
        upper = str(action).upper()
        op = upper if upper in OPS else None
    if op is None:
        return "[决策失败 ✗]"
    brief = (args.get("text") or args.get("id") or args.get("element_id")
             or args.get("clicks") or "")
    return f"[{op} {brief} {'✓' if ok else '✗'}]"


def load_engine(ckpt: str):
    """懒加载 Jev 推理引擎（唯一 import torch 的地方）。

    加载失败（未装 torch、显存不足、ckpt 损坏）返回 None：
    run_task 自动退化成纯 VLM 循环，不能让加速组件把主流程拖挂。
    """
    import logging

    try:
        from training.serve import JevDecision

        return JevDecision(ckpt)
    except Exception as e:  # noqa: BLE001 - 任何加载失败都只降级，不抛
        logging.getLogger(__name__).warning("Jev 引擎加载失败，回退纯 VLM 循环: %s", e)
        return None
