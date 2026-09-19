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

import re

# ---- 格式版本 ----
# v1：没有状态位（2026-09-19 之前采的所有轨迹）。此时"元素行里没有 flag" = **状态未知**。
# v2：采到了状态位。此时"没有 flag" = **明确为 false**。
# 这两种语义必须可区分：模型对"没这个字段"和"字段为空"的理解不一样，混用就是训练噪声。
# 代价：goal 行多一个字符（约 1 token），换来整个数据集口径唯一。
FORMAT_VERSION = 2

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


_STATE_KEYS = ("disabled", "checked", "expanded", "selected")


def flags_of(el: dict) -> str:
    """元素的紧凑状态串。

    三种情形必须分开（"未知 ≠ false ≠ 不适用"）：
    - 元素里**一个状态字段都没有** → v1 老记录，整条数据都没状态位，由首行版本号说明
    - 四个状态位**全是 None** → 整个元素读不到（窗口关了/控件没了）→ 打 `?`
    - 其余 → 只写为 true 的那些字母。**没写的既包含 false，也包含"该控件不支持这种
      状态"**（按钮的勾选、文本框的展开都属此类；实测按钮不支持 TogglePattern，
      若把"不适用"也当未知，66 个元素会全部变成 `?`）
    """
    if not any(key in el for key in _STATE_KEYS):
        return ""
    if all(el.get(key) is None for key in _STATE_KEYS):
        return "?"
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
                          foreground: str = "", version: int = FORMAT_VERSION) -> str:
    """一个决策步的完整文本输入（**不含动作说明书**——那是权重里的东西）。

    结构：版本 + goal / window / 最近 k 步历史 / 元素表。
    首行的 `T2` 是格式版本：没有数字（v1 老数据）表示**状态位未知**，
    有 `2` 表示元素行里的 flag 是权威的（没写就是 false）。见 FORMAT_VERSION。
    """
    parts = [f"T{version} {goal}"]
    if foreground:
        parts.append(f"W {foreground}")
    parts.append(f"H {serialize_history(history or [], k_history)}")
    parts.append(serialize_elements(elements, style))
    return "\n".join(parts)


# ==================== 会话级持久 id ====================
# 为什么必须有它（增量编码的前置依赖）：
#   1. 增量编码第 5 步只发"变化的 10-20 个元素"，模型要点的元素可能是几步前发的 →
#      id 必须在整个任务会话内稳定，不能每步重编号；
#   2. 历史里 `7C #42` 指向的是若干步前的快照，id 漂移 = 历史全是坏引用；
#   3. SoM 图上的编号、文本表里的 id、动作输出的目标，三者必须共用同一套 id 空间。
# 晚改的代价：已训练数据里的 id 全部失效，等于全量重生成。

def element_fingerprint(el: dict) -> tuple:
    """元素的会话内指纹（决定它是不是"同一个控件"）。

    首选 UIA RuntimeId：它就是 UIA 自己用来跨调用认元素的东西，最稳。
    RuntimeId 拿不到时退化为 (进程, 窗口, 类型, 名字, 粗粒度 bbox)：
    bbox 取整到 8px 是为了容忍细微位移，同时把"同窗口同名的多个控件"区分开。
    """
    rid = el.get("runtime_id")
    if rid:
        return ("r", tuple(int(v) for v in rid))
    bbox = el.get("bbox") or [0, 0, 0, 0]
    return ("f", int(el.get("pid", 0)), int(el.get("window_handle", 0)),
            str(el.get("control_type", "")), str(el.get("name", ""))[:32],
            tuple(int(v) // 8 for v in bbox))


class SessionIds:
    """一次任务会话内的元素 id 分配器：同一个控件恒定拿到同一个 id。

    - id 只增不复用（元素消失后它的 id 仍然属于它，历史引用不会指向别人）
    - 每步能报出"新出现的 id"和"消失的 id" → 增量编码直接用
    """

    def __init__(self) -> None:
        self._by_fp: dict[tuple, int] = {}
        self._next = 1
        self._alive: set[int] = set()

    def assign(self, elements: list[dict]) -> dict:
        """就地给元素写 `id`（覆盖采集器的每步重编号），返回本步的变化摘要。

        返回 {"ids": [...], "new": [...], "gone": [...]}
        """
        ids, new = [], []
        seen: set[int] = set()
        for el in elements:
            fp = element_fingerprint(el)
            eid = self._by_fp.get(fp)
            if eid is None:
                eid = self._next
                self._next += 1
                self._by_fp[fp] = eid
                new.append(eid)
            el["id"] = eid
            ids.append(eid)
            seen.add(eid)
        gone = sorted(self._alive - seen)
        self._alive = seen
        return {"ids": ids, "new": new, "gone": gone}

    def forget(self, ids: list[int]) -> None:
        """删除若干 id 的指纹登记（会话结束或强制全量同步后清理用）。"""
        drop = set(ids)
        self._by_fp = {fp: i for fp, i in self._by_fp.items() if i not in drop}
        self._alive -= drop


# ==================== goal 风格规范 ====================
# 训练用的 goal 必须和**未来 System 2 规划器产出的子目标**同风格，否则又是一次
# train/serve skew：训练时是"把当前文档保存到桌面"，部署时规划器给"保存"。
MAX_GOAL_CHARS = 15
_COORD_RE = re.compile(r"\d+\s*[,，]\s*\d+")
_ELEM_REF_RE = re.compile(r"#\s*\d+|编号\s*\d*")


def check_goal_style(goal: str) -> str | None:
    """goal 风格检查：合规返回 None，否则返回原因。

    规范（2026-09-19 定）：中文短祈使句、≤15 字、不含坐标与元素编号。
    教师生成与模板生成都必须过这一关——**目标是学生学到的唯一任务信号**，
    写歪了整批数据都歪。
    """
    text = str(goal).strip()
    if not text:
        return "goal 为空"
    if len(re.sub(r"\s", "", text)) > MAX_GOAL_CHARS:
        return f"超过 {MAX_GOAL_CHARS} 字（现状 {len(text)}）"
    if _COORD_RE.search(text):
        return "含坐标：坐标只存在于 SoM 图里"
    if _ELEM_REF_RE.search(text):
        return "含元素编号：等于把答案写进题面"
    return None
