"""观测序列化标准（单一真源）——训练、推理、对照台共用同一份。

**格式决策锚定路线 B（头手术后）的终局形态**，不是零样本 API 模型：

- 动作空间长在权重里 → prompt 里**不写动作说明书**（旧 `_LOOP_PROMPT` 那 623 token
  的中文动作清单是"提示词时代的税"，微调后一个字都不用交）
- 不使用坐标：坐标只存在于 SoM 图像里。文本里给 bbox 不但费 token
  （实测每元素 26.7 vs 10.3），还给模型一条"无视 SoM 直接回归坐标"的捷径，
  训练时和视觉分支抢梯度，两头都学不精
- 保留 ControlType（**免费的语义先验**：按钮 vs 文字）与状态位（disabled/checked/
  expanded/selected 对"先选中才能删"这类决策是必需信息），代价约 1 token/元素

预算目标（2026-09-19 实测基线 2188 token/步）：

| 形态 | 首步 | 后续步（前缀缓存命中） |
|---|---|---|
| 路线 B 终局 | ~770 | <400 |

这里只负责**格式**；token 与准确率的 Pareto 由 `scripts/token_budget.py` 与对照台量。
"""

from __future__ import annotations

# ---- 控件类型 → 单字母（UIA 的类型名很长：ButtonControl = 14 字符 ≈ 4-5 token）----
TYPE_CODE = {
    "ButtonControl": "B",
    "SplitButtonControl": "B",
    "EditControl": "I",          # input
    "CheckBoxControl": "C",
    "RadioButtonControl": "R",
    "ComboBoxControl": "X",      # 下拉
    "ListItemControl": "L",
    "TreeItemControl": "T",
    "DataItemControl": "D",      # 表格单元格
    "MenuItemControl": "M",
    "TabItemControl": "K",
    "DocumentControl": "W",
    "ListControl": "G",          # 容器（group/list 本身）
    "TreeControl": "G",
    "TableControl": "G",
    "PaneControl": "G",
    "WindowControl": "W",
}

# ---- 状态位：只在真有状态时追加，没状态就一个字符都不花 ----
# d=禁用  c=勾选  e=展开  s=选中  f=有焦点
FLAG_CODE = (("disabled", "d"), ("checked", "c"), ("expanded", "e"),
             ("selected", "s"), ("focused", "f"))

# ---- 动作 → 单字母（历史压缩用；与 jev_bridge.OPS 同集合）----
OP_CODE = {
    "CLICK": "C", "DOUBLE_CLICK": "D", "RIGHT_CLICK": "R", "TYPE": "T",
    "SCROLL": "S", "WAIT": "W", "DONE": "F", "BLOCKED": "X",
}
_CODE_OP = {v: k for k, v in OP_CODE.items()}

MAX_NAME_CHARS = 16


def type_code(control_type: str) -> str:
    return TYPE_CODE.get(control_type, "?")


def flags_of(el: dict) -> str:
    """元素的紧凑状态串（无状态则空串）。"""
    return "".join(letter for key, letter in FLAG_CODE if el.get(key))


def format_element(el: dict, style: str = "tight") -> str:
    """一个元素 → 一行文本。

    style:
      tight  ← **推荐/训练标准**：id + 类型 + 状态 + 名字
      min    最瘦：id + 名字（砍掉类型，实测每元素省 ~1.2 token，但要 Pareto 说了算）
      legend 现状（id: [全类型名] 名字）——只为对照台保留
      bbox   旧的 Jev 格式（含坐标）——**反例**，只为对照台保留
    """
    name = str(el.get("name", ""))[:MAX_NAME_CHARS]
    if style == "min":
        return f"{el['id']}:{name or '-'}"
    if style == "legend":
        return f"{el['id']}: [{el.get('control_type', '')}] {name or '(无名)'}"
    if style == "bbox":
        bbox = " ".join(str(int(v)) for v in el.get("bbox", [0, 0, 0, 0]))
        return f"<|elem|>#{el['id']} {el.get('control_type', '')} \"{name}\" {bbox}"
    # tight
    tag = f"{el['id']}{type_code(el.get('control_type', ''))}{flags_of(el)}"
    return f"{tag} {name or '-'}"


def serialize_elements(elements: list[dict], style: str = "tight") -> str:
    return "\n".join(format_element(e, style) for e in elements)


def format_history_entry(step_idx: int, op: str, elem: dict | None = None,
                         text: str = "", ok: bool = True) -> str:
    """历史一步 → 单行（约 6-10 token）。

    例：`7C #42`  第 7 步 CLICK 42 号元素
        `8T #12 "78*9"`  第 8 步在 12 号元素输入文本
    """
    code = OP_CODE.get(op, "?")
    bits = [f"{step_idx}{code}"]
    if elem is not None:
        bits.append(f"#{elem.get('id')}")
    if text:
        bits.append(f"\"{str(text)[:24]}\"")
    if not ok:
        bits.append("!")          # 该步执行失败（不是成功动作）
    return " ".join(bits)


def serialize_history(entries: list[str], k: int = 4) -> str:
    """只保留最近 k 步（DOM 决策本质是马尔可夫的；长历史既费 token 又引入噪声）。

    `entries` 是 `format_history_entry` 产出的单行串。
    """
    if not entries:
        return "-"
    return "\n".join(entries[-k:])


def serialize_observation(goal: str, elements: list[dict],
                          history: list[str] | None = None,
                          style: str = "tight", k_history: int = 4,
                          foreground: str = "") -> str:
    """一个决策步的完整文本输入（**不含动作说明书**——那是权重里的东西）。

    结构：goal / window / 最近 k 步历史 / 元素表。
    """
    parts = [f"T {goal}"]
    if foreground:
        parts.append(f"W {foreground}")
    parts.append(f"H {serialize_history(history or [], k_history)}")
    parts.append(serialize_elements(elements, style))
    return "\n".join(parts)
