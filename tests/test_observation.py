"""观测格式标准（`screen_use/observation.py`）的护栏。

这些断言的价值不在"格式对不对"，而在**别再把格式改回去**：
2026-09-19 实测过，旧格式（中文动作说明书 + 全类型名 legend + 逐元素 bbox）
每步 1891 token，其中大部分是"提示词时代的税"——微调后一个字都不用交。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from screen_use import observation as ob  # noqa: E402

ELS = [
    {"id": 1, "name": "保存", "control_type": "ButtonControl", "bbox": [0, 0, 10, 10]},
    {"id": 2, "name": "文件名", "control_type": "EditControl", "bbox": [0, 20, 10, 30]},
    {"id": 3, "name": "删除", "control_type": "ButtonControl", "disabled": True,
     "bbox": [0, 40, 10, 50]},
    {"id": 4, "name": "", "control_type": "TreeItemControl", "expanded": True,
     "bbox": [0, 60, 10, 70]},
]


def test_type_codes_are_one_letter_and_cover_real_types():
    """真实轨迹里出现过的类型都要有单字母码，否则会退化成 '?' 白花 token。"""
    seen = {"ButtonControl", "SplitButtonControl", "TabItemControl", "EditControl",
            "DocumentControl", "MenuItemControl", "DataItemControl", "ListItemControl",
            "TreeItemControl", "CheckBoxControl", "RadioButtonControl"}
    for ctype in seen:
        code = ob.type_code(ctype)
        assert len(code) == 1 and code != "?", f"{ctype} 没有单字母码"


def test_flags_cost_nothing_when_absent():
    """没有状态就不该多花一个字符——90 个元素里 80 个是无状态的。"""
    plain = ob.format_element({"id": 7, "name": "确定", "control_type": "ButtonControl"})
    assert plain == "7B 确定"

    off = ob.format_element({"id": 7, "name": "确定", "control_type": "ButtonControl",
                             "disabled": True})
    assert off == "7Bd 确定"


def test_tight_keeps_type_and_state_min_strips_them():
    """④b 保留语义先验与状态位；④ 最瘦但砍掉了它们——两者都要能出，交给 Pareto 定。"""
    tight = ob.format_element(ELS[2], "tight")
    assert tight.startswith("3Bd "), tight          # 类型 + 禁用状态都在
    assert ob.format_element(ELS[2], "min") == "3:删除"


def test_bbox_style_is_marked_as_the_counterexample():
    """旧 Jev 格式带坐标：既费 token（实测 26.7/元素），又给模型"无视 SoM 回归坐标"
    的捷径。它只该出现在对照台里，不该是训练标准。"""
    line = ob.format_element(ELS[0], "bbox")
    assert "<|elem|>" in line and "0 0 10 10" in line
    assert ob.format_element(ELS[0], "tight") == "1B 保存"


def test_unnamed_controls_do_not_produce_empty_field():
    """无名图标控件占 10%（真实实测）——留空会让模型以为格式坏了，给个占位符。"""
    assert ob.format_element(ELS[3], "tight") == "4Te -"
    assert ob.format_element(ELS[3], "min") == "4:-"


def test_history_keeps_only_last_k_and_is_one_line_each():
    entries = [ob.format_history_entry(i, "CLICK", {"id": 40 + i}) for i in range(1, 9)]

    out = ob.serialize_history(entries, k=4)

    assert out.splitlines() == entries[-4:]
    assert len(out.splitlines()) == 4
    assert out.splitlines()[-1] == "8C #48"       # 单行、无中文、无思考文本


def test_history_marks_failed_steps_without_exploding_cost():
    ok = ob.format_history_entry(3, "CLICK", {"id": 5}, ok=True)
    bad = ob.format_history_entry(3, "CLICK", {"id": 5}, ok=False)

    assert ok == "3C #5" and bad == "3C #5 !"
    assert len(bad) - len(ok) <= 2


def test_observation_contains_no_action_manual():
    """核心决策：动作空间长在权重里，prompt 里不写说明书。

    旧 `_LOOP_PROMPT` 里那段中文动作清单是给零样本模型看的（实测 279 token/步）。
    这条断言防止它被顺手加回来。
    """
    text = ob.serialize_observation("算 78 乘以 9", ELS[:2],
                                    [ob.format_history_entry(1, "CLICK", {"id": 1})])
    for leaked in ("action", "click_id", "thought", "你觉得", "可用动作", "JSON"):
        assert leaked not in text, f"观测里混进了动作说明书：{leaked}"


def test_observation_layout_is_goal_window_history_elements():
    text = ob.serialize_observation("存盘", ELS[:1], [], foreground="记事本")

    lines = text.splitlines()
    assert lines[0] == "T 存盘"
    assert lines[1] == "W 记事本"
    assert lines[2] == "H -"            # 无历史
    assert lines[3] == "1B 保存"
