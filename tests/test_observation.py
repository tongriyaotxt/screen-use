"""观测格式标准（`screen_use/observation.py`）的护栏。

这些断言的价值不在"格式对不对"，而在**别再把格式改回去**：
2026-09-19 实测过，旧格式（中文动作说明书 + 全类型名 legend + 逐元素 bbox）
每步 1891 token，其中大部分是"提示词时代的税"——微调后一个字都不用交。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from screen_use import observation as ob  # noqa: E402

def _v2(eid, name, ctype, bbox, **state):
    """一个**格式 v2**的元素：四个状态位齐全（None = 这一项读取失败）。

    真实采集器（`uia_tree._annotate_state` → `UIElement.to_dict`）永远写全四个键，
    所以这才是数据的真实形状；只给部分字段的 dict 属于历史遗留。
    """
    el = {"id": eid, "name": name, "control_type": ctype, "bbox": list(bbox),
          "disabled": False, "checked": False, "expanded": False, "selected": False}
    el.update(state)
    return el


ELS = [
    _v2(1, "保存", "ButtonControl", (0, 0, 10, 10)),
    _v2(2, "文件名", "EditControl", (0, 20, 10, 30)),
    _v2(3, "删除", "ButtonControl", (0, 40, 10, 50), disabled=True),
    _v2(4, "", "TreeItemControl", (0, 60, 10, 70), expanded=True),
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
    """没有状态就不该多花一个字符——90 个元素里绝大多数是无状态的。"""
    plain = ob.format_element(_v2(7, "确定", "ButtonControl", (0, 0, 1, 1)))
    assert plain == "7B 确定"

    off = ob.format_element(_v2(7, "确定", "ButtonControl", (0, 0, 1, 1), disabled=True))
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


def test_flags_distinguish_unknown_from_false():
    """三种情形必须给出三种不同的编码——这是"未知 ≠ false"的落点。"""
    v1 = {"id": 1, "name": "保存", "control_type": "ButtonControl"}
    known = {"id": 1, "name": "保存", "control_type": "ButtonControl",
             "disabled": False, "checked": False, "expanded": False, "selected": False}
    dead = {"id": 1, "name": "保存", "control_type": "ButtonControl",
            "disabled": None, "checked": None, "expanded": None, "selected": None}

    assert ob.format_element(v1) == "1B 保存"        # 老数据：整条由版本号说明
    assert ob.format_element(known) == "1B 保存"     # 明确都没状态
    assert ob.format_element(dead) == "1B? 保存"     # 元素读不到 → 如实标未知


def test_pattern_not_applicable_is_false_not_unknown():
    """按钮不支持 TogglePattern → 它的"勾选"是**不适用**，不是未知。

    现场实测踩过：把"不适用"当未知，66 个元素会全部序列化成 `?`，
    状态位等于白采，还把数据污染成"什么都拿不准"。
    """
    button = _v2(1, "保存", "ButtonControl", (0, 0, 1, 1))   # checked=False 即不适用
    assert ob.flags_of(button) == "" and ob.format_element(button) == "1B 保存"

    tree = _v2(2, "文档", "TreeItemControl", (0, 0, 1, 1), expanded=True)
    assert ob.format_element(tree) == "2Te 文档"


def test_legacy_data_can_be_serialized_as_v1():
    """老轨迹落盘时没有状态位，重新序列化要能如实标 v1，不能冒充 v2。"""
    v1 = {"id": 1, "name": "保存", "control_type": "ButtonControl"}

    text = ob.serialize_observation("存盘", [v1], version=1)

    assert text.startswith("T1 ")


def test_observation_layout_is_goal_window_history_elements():
    text = ob.serialize_observation("存盘", ELS[:1], [], foreground="记事本")

    lines = text.splitlines()
    assert lines[0] == "T2 存盘"          # T2 = 格式 v2（带状态位口径）
    assert lines[1] == "W 记事本"
    assert lines[2] == "H -"              # 无历史
    assert lines[3] == "1B 保存"


# ---- 格式版本：状态位"未知"与"明确为 false"必须可区分 ----

def test_format_version_separates_unknown_state_from_false_state():
    """v1 老数据没有状态位 → 元素行里的"没写 flag"是**未知**，不是 false。

    微调时把两者混为一谈 = 训练噪声；这条测试把这个口径钉死。
    """
    assert ob.FORMAT_VERSION == 2
    text = ob.serialize_observation("存盘", ELS[:1])
    assert text.startswith(f"T{ob.FORMAT_VERSION} "), "新数据必须带版本号"
    assert "T1 " not in text


# ---- 会话级持久 id：增量编码的前置依赖 ----

def _el(eid, name, ctype="ButtonControl", win=1, rid=None, bbox=(0, 0, 10, 10)):
    el = {"id": eid, "name": name, "control_type": ctype, "window_handle": win,
          "bbox": list(bbox)}
    if rid is not None:
        el["runtime_id"] = rid
    return el


def test_ids_survive_reordering_across_steps():
    """同一批元素换个顺序枚举，id 必须一模一样——否则历史引用全成坏引用。"""
    ids = ob.SessionIds()
    els = [_el(1, "保存", rid=[1, 10]), _el(2, "取消", rid=[1, 11]),
           _el(3, "文件名", "EditControl", rid=[1, 12])]

    first = ids.assign([dict(e) for e in els])["ids"]
    shuffled = ids.assign([dict(e) for e in reversed(els)])["ids"]

    assert first != shuffled                      # 顺序确实变了
    assert sorted(first) == sorted(shuffled)      # 但 id 集合不变
    by_name = {e["name"]: e["id"] for e in els}
    assert by_name["保存"] == first[0] and by_name["文件名"] == first[2]


def test_id_is_never_reused_after_element_disappears():
    """元素消失后它的 id 仍属于它：历史里的 #42 不能指向后来别的控件。"""
    ids = ob.SessionIds()
    a = _el(1, "保存", rid=[1, 10])
    b = _el(2, "取消", rid=[1, 11])
    saved = ids.assign([a, b])["ids"][0]

    gone = ids.assign([b])                        # 保存 消失
    assert gone["gone"] == [saved]

    fresh = _el(9, "新按钮", rid=[1, 99])
    step = ids.assign([b, fresh])
    assert fresh["id"] not in (saved, b["id"]), "新元素不能复用已消失元素的 id"


def test_new_and_gone_are_reported_for_delta_encoding():
    """增量编码就靠这两个集合：只发 new（含变化的），再告诉模型 gone 了哪些。"""
    ids = ob.SessionIds()
    a, b = _el(1, "保存", rid=[1, 10]), _el(2, "取消", rid=[1, 11])
    first = ids.assign([a, b])
    assert len(first["new"]) == 2 and first["gone"] == []

    c = _el(3, "确定", rid=[1, 12])
    second = ids.assign([a, c])                   # b 走了，c 来了
    assert second["new"] == [c["id"]]
    assert second["gone"] == [b["id"]]


def test_fingerprint_falls_back_when_runtime_id_missing():
    """老采集器没有 RuntimeId → 退化为 (进程, 窗口, 类型, 名字, 粗 bbox)。"""
    ids = ob.SessionIds()
    el = _el(1, "保存")
    el.pop("runtime_id", None)
    first = ids.assign([el])["ids"][0]

    same = _el(7, "保存")                          # 同名同窗口同位置，只是每步重编号
    same.pop("runtime_id", None)
    assert ids.assign([same])["ids"][0] == first


def test_same_name_in_different_windows_gets_different_ids():
    """两个窗口都有「确定」——它们必须是两个元素，否则点了另一个窗口的按钮。"""
    ids = ob.SessionIds()
    w1, w2 = _el(1, "确定", win=1), _el(2, "确定", win=2)
    for e in (w1, w2):
        e.pop("runtime_id", None)

    got = ids.assign([w1, w2])["ids"]

    assert len(set(got)) == 2


def test_runtime_id_wins_over_position_change():
    """元素被挪了位置（bbox 变了）也不该换 id——RuntimeId 优先就是为这个。"""
    ids = ob.SessionIds()
    el = _el(1, "保存", rid=[1, 10], bbox=(0, 0, 10, 10))
    first = ids.assign([el])["ids"][0]

    moved = _el(5, "保存", rid=[1, 10], bbox=(300, 400, 310, 410))
    assert ids.assign([moved])["ids"][0] == first


# ---- goal 风格规范（防止又一次 train/serve skew） ----

def test_goal_style_rejects_coordinates_and_element_refs():
    assert ob.check_goal_style("保存文件") is None
    assert ob.check_goal_style("算 78 乘以 9") is None          # 数字本身合法
    assert "坐标" in ob.check_goal_style("点击 (100, 200)")
    assert "编号" in ob.check_goal_style("点击 #42")
    assert "15" in ob.check_goal_style("把当前这个文档保存到桌面上的文件夹里面去")
