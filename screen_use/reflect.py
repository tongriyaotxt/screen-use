"""困难分类与应对提示词库（introspection playbook）。

不同困难类型给 VLM 不同的引导提示，比通用反思更有效：

| 困难类型 | 检测信号 | 提示策略 |
|---|---|---|
| 动作无效 | 截图前后无变化 | 检查焦点/激活窗口/换双击 |
| 重复循环 | 同一动作连续 3 次 | 强制换路径，禁止重复 |
| 连续失败 | 连续 2 步执行报错 | 换交互方式（菜单/快捷键） |
| 元素缺失 | 候选列表为空 | 等待/滚动/切换窗口 |
| 意外弹窗 | 前台窗口标题突变 | 优先处理弹窗 |
| 通用兜底 | 其他 | 自由分析 |
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class StuckType(str, Enum):
    NO_EFFECT = "no_effect"            # 动作执行成功但屏幕没变化
    REPEAT_LOOP = "repeat_loop"        # 同一动作连续重复
    CONSECUTIVE_FAILURES = "failures"  # 连续执行失败
    NO_ELEMENTS = "no_elements"        # 找不到可交互元素
    POPUP = "popup"                    # 疑似意外弹窗
    GENERIC = "generic"


_TEMPLATES: dict[StuckType, str] = {
    StuckType.NO_EFFECT: """【困难：动作无效】动作报告成功，但屏幕没有任何变化。
引导思路：
1. 目标窗口是否真的获得了焦点？（可先点击窗口标题栏或正文区域激活）
2. 点击位置是否被其他窗口/弹窗遮挡？
3. 这个控件是否需要双击、右键，或先悬停？
4. 应用是否在加载中，需要 wait 后重试？""",

    StuckType.REPEAT_LOOP: """【困难：重复循环】同一个动作已经连续执行了 3 次，说明当前策略无效。
强制要求：
1. 禁止再执行相同的动作和参数
2. 必须换一种方式：例如从点击换成快捷键、从找按钮换成菜单、或直接操作坐标
3. 如果目标在当前界面找不到，考虑滚动页面或切换选项卡""",

    StuckType.CONSECUTIVE_FAILURES: """【困难：连续失败】最近两步动作都执行失败了。
引导思路：
1. 失败原因是元素失效（界面已变化）→ 重新观察当前候选元素列表
2. 失败原因是坐标/ID 错误 → 改用其他动作类型
3. 考虑用系统级替代路径：右键菜单、快捷键、或从菜单栏进入""",

    StuckType.NO_ELEMENTS: """【困难：找不到可交互元素】当前候选元素列表为空或没有目标。
引导思路：
1. 目标应用可能不在前台 → 用 hotkey alt+tab 切换窗口
2. 应用可能还在加载 → wait 1-2 秒后再观察
3. 目标可能在可视区域外 → scroll 滚动查找
4. 该程序可能是自绘界面（UIA 盲区）→ 改用 click 坐标动作直接点""",

    StuckType.POPUP: """【困难：疑似意外弹窗】前台窗口突然变化，可能有弹窗遮挡了操作。
引导思路：
1. 先观察截图判断弹窗内容：更新提示？确认对话框？广告？
2. 优先处理弹窗：点击"取消/关闭/稍后"等按钮，或按 esc
3. 处理完弹窗后，回到原任务继续""",

    StuckType.GENERIC: """【困难：进展停滞】任务没有明显进展。
引导思路：重新审视任务目标，将剩余步骤拆得更细，逐步验证。""",
}

_REFLECT_SUFFIX = """
请结合截图分析，只输出 JSON：
{{"analysis": "一句话说明卡住的根本原因", "strategy": "下一步具体怎么做（含建议的动作类型）"}}"""


@dataclass
class StuckSignal:
    """卡住检测的输入信号。"""

    steps: list[Any]            # 最近的 Step 列表
    screen_changed: bool        # 最近一步执行后屏幕是否有变化
    elements_count: int         # 当前可交互元素数量
    window_title_changed: bool  # 前台窗口标题是否突变


def screen_fingerprint(image_base64: str) -> str:
    """截图指纹：直接对 base64 内容取 hash（同尺寸同内容即同指纹）。

    简单可靠：用于检测"动作执行了但屏幕没变化"。
    """
    return hashlib.md5(image_base64.encode()).hexdigest()


def classify_stuck(signal: StuckSignal) -> StuckType | None:
    """根据信号分类困难类型。未卡住返回 None。

    优先级：重复循环 > 连续失败 > 动作无效 > 元素缺失 > 弹窗
    """
    steps = signal.steps

    # 重复循环：同一动作+参数连续 3 次
    if len(steps) >= 3:
        sigs = [
            (s.action, json.dumps(s.args, sort_keys=True, ensure_ascii=False))
            for s in steps[-3:]
        ]
        if sigs[0] == sigs[1] == sigs[2]:
            return StuckType.REPEAT_LOOP

    # 连续失败：最近 2 步都执行报错
    if len(steps) >= 2 and not steps[-1].ok and not steps[-2].ok:
        return StuckType.CONSECUTIVE_FAILURES

    # 动作无效：上一步动作（其效果体现在本轮截图）成功但屏幕毫无变化
    if (
        len(steps) >= 2
        and steps[-2].ok
        and steps[-2].action not in ("wait",)
        and not signal.screen_changed
    ):
        return StuckType.NO_EFFECT

    # 元素缺失
    if signal.elements_count == 0:
        return StuckType.NO_ELEMENTS

    # 意外弹窗
    if signal.window_title_changed:
        return StuckType.POPUP

    return None


def build_reflect_prompt(stuck_type: StuckType, goal: str, history: list[str]) -> str:
    """按困难类型生成针对性的反思提示词。"""
    template = _TEMPLATES.get(stuck_type, _TEMPLATES[StuckType.GENERIC])
    return f"""你正在执行的桌面任务遇到了困难，请做一次内省反思。

任务：{goal}

已执行的动作历史：
{chr(10).join(history)}

{template}{_REFLECT_SUFFIX}"""
