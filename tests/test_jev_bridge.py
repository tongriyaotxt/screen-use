"""Jev 桥接层测试：动作词表单一真源、决策翻译、history 与训练侧同构。"""

import pytest

from screen_use import jev_bridge


def test_ops_single_source_of_truth():
    """训练侧词表必须就是运行时那一份（防 train/serve 漂移）。"""
    from training import common

    assert common.OPS is jev_bridge.OPS
    assert common.OP2IDX == jev_bridge.OP2IDX
    assert len(jev_bridge.OPS) == 8


def test_every_op_has_action_mapping():
    assert set(jev_bridge.JEV_TO_ACTION) == set(jev_bridge.OPS)
    # 除 BLOCKED（认输）外都有可执行动作
    assert jev_bridge.JEV_TO_ACTION["BLOCKED"] is None


# ---- to_action ----

def test_to_action_click():
    d = jev_bridge.to_action({"action": "CLICK", "element_id": 7, "confidence": 0.9})
    assert d["action"] == "click_id" and d["id"] == 7
    assert "conf=0.90" in d["thought"]


@pytest.mark.parametrize("op,action", [
    ("DOUBLE_CLICK", "double_click"), ("RIGHT_CLICK", "right_click"),
])
def test_to_action_mouse_variants(op, action):
    d = jev_bridge.to_action({"action": op, "element_id": 3, "confidence": 0.8})
    assert d["action"] == action and d["id"] == 3


def test_to_action_type_carries_text_and_target():
    d = jev_bridge.to_action({"action": "TYPE", "element_id": 4,
                              "text": "北京", "confidence": 0.8})
    assert d == {"action": "type", "thought": "jev:TYPE conf=0.80",
                 "text": "北京", "id": 4}


def test_to_action_type_without_text_is_not_executable():
    """span 抄不到文本 → 返回 None（升级），绝不编文本（TRAINING §1）。"""
    assert jev_bridge.to_action({"action": "TYPE", "element_id": 4,
                                 "text": None, "confidence": 0.9}) is None


def test_to_action_element_op_without_element_is_not_executable():
    assert jev_bridge.to_action({"action": "CLICK", "element_id": None}) is None


def test_to_action_blocked_is_not_executable():
    assert jev_bridge.to_action({"action": "BLOCKED", "confidence": 1.0}) is None


def test_to_action_unknown_op_is_not_executable():
    assert jev_bridge.to_action({"action": "HOTKEY", "confidence": 1.0}) is None
    assert jev_bridge.to_action({}) is None


def test_to_action_defaults_for_payloadless_ops():
    """v0 动作集没有格数/秒数头 → 用默认载荷（DESIGN §5.1 已知精度损失）。"""
    scroll = jev_bridge.to_action({"action": "SCROLL", "confidence": 0.9})
    assert scroll["clicks"] == jev_bridge.DEFAULT_SCROLL_CLICKS
    wait = jev_bridge.to_action({"action": "WAIT", "confidence": 0.9})
    assert wait["seconds"] == jev_bridge.DEFAULT_WAIT_SECONDS
    done = jev_bridge.to_action({"action": "DONE", "confidence": 0.9})
    assert done["action"] == "done" and done["result"]


def test_to_action_done_keeps_given_result():
    d = jev_bridge.to_action({"action": "DONE", "result": "已删除", "confidence": 0.9})
    assert d["result"] == "已删除"


# ---- history 与训练侧同构 ----

@pytest.mark.parametrize("action,args", [
    ("click_id", {"id": 12}),
    ("type", {"text": "张三", "element_id": 3}),
    ("scroll", {"clicks": -8}),
    ("done", {"result": "已删除"}),
    ("wait", {}),
])
def test_history_entry_matches_training_side(action, args):
    """运行时 history 必须与训练侧同一实现（否则 serve 分布漂移）。"""
    from training.common import _hist_entry

    op = jev_bridge.ACTION_TO_JEV[action]
    assert (jev_bridge.history_entry(action, args, True)
            == _hist_entry(op, {"args": args}, True))


@pytest.mark.parametrize("action,args,ok,expect", [
    ("click_id", {"id": 12}, True, "[CLICK 12 ✓]"),
    ("type", {"text": "张三", "element_id": 3}, True, "[TYPE 张三 ✓]"),
    ("scroll", {"clicks": -8}, False, "[SCROLL -8 ✗]"),
    ("done", {"result": "已删除"}, True, "[DONE  ✓]"),   # done 不带载荷（与训练一致）
    ("wait", {}, True, "[WAIT  ✓]"),
    ("blocked", {"reason": "元素不存在"}, True, "[BLOCKED  ✓]"),
])
def test_history_entry_literal_format(action, args, ok, expect):
    """格式钉死：这些字符串就是 SFT 训练时喂给模型的 history（改了就等于换分布）。"""
    assert jev_bridge.history_entry(action, args, ok) == expect


def test_history_entry_accepts_jev_op_too():
    assert jev_bridge.history_entry("CLICK", {"id": 5}, False) == "[CLICK 5 ✗]"


def test_history_entry_maps_click_coordinate_to_click():
    """坐标点击在训练动作集里也是 CLICK（动作词表不含坐标）。"""
    assert jev_bridge.history_entry("click", {"x": 1, "y": 2}, True) == "[CLICK  ✓]"


@pytest.mark.parametrize("action", ["hotkey", "press", "think_error"])
def test_history_entry_marks_out_of_set_actions_as_failure(action):
    """动作集外的步按训练侧约定记作「决策失败 ✗」（分布必须与 SFT 一致）。"""
    from training.common import _hist_entry

    assert jev_bridge.history_entry(action, {"keys": "ctrl+s"}, True) == "[决策失败 ✗]"
    assert _hist_entry(action, {"args": {}}, True) == "[决策失败 ✗]"


# ---- 训练分布门控（观测尺度远超训练集 → 别问了，直接升级） ----

_CALIB = {"train_stats": {"elements_p50": 14, "elements_p95": 20, "elements_max": 25}}


def test_ood_check_within_distribution():
    assert jev_bridge.ood_check(14, _CALIB) is None
    assert jev_bridge.ood_check(25, _CALIB) is None      # 训练集最大值仍算分布内
    assert jev_bridge.ood_check(20, _CALIB) is None      # 长尾也交给置信度头，不重复收税


def test_ood_check_out_of_distribution():
    """真实桌面常见 80+ 元素 → 判定分布外，省掉一次必然被推翻的前向。"""
    reason = jev_bridge.ood_check(84, _CALIB)
    assert reason and "超出训练分布" in reason and "84" in reason


def test_ood_check_without_stats():
    """旧 ckpt 没落盘训练统计 → 不做门控（行为回退到只看置信度）。"""
    assert jev_bridge.ood_check(999, {}) is None
    assert jev_bridge.ood_check(999, {"train_stats": {}}) is None
    assert jev_bridge.ood_check(999, None) is None


# ---- 引擎懒加载降级 ----

def test_load_engine_degrades_when_engine_raises(monkeypatch):
    """引擎加载失败（无 torch / 显存不足 / ckpt 损坏）→ None，不把主流程拖挂。

    这里用假模块注入失败，避免为了测降级真的 import torch（测试要快）。
    """
    import sys
    import types

    fake = types.ModuleType("training.serve")

    def boom(ckpt):
        raise RuntimeError("CUDA out of memory")

    fake.JevDecision = boom
    monkeypatch.setitem(sys.modules, "training.serve", fake)
    assert jev_bridge.load_engine("whatever") is None
