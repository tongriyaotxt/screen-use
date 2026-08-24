"""视觉行为循环测试：parse_action + 循环执行（mock provider，dry_run）。"""

from unittest.mock import patch

import pytest

from screen_use.agent import VisualLoop, parse_action
from screen_use.config import Settings
from screen_use.perception.uia_tree import UIElement
from screen_use.providers.base import VisionProvider
from screen_use.tools import ScreenUse

FAKE_ELEMENTS = [
    UIElement(id=1, name="One", control_type="ButtonControl", bbox=(0, 0, 50, 50)),
]


# ---- parse_action ----

def test_parse_action_plain():
    d = parse_action('{"thought": "点1", "action": "click_id", "id": 1}')
    assert d["action"] == "click_id" and d["id"] == 1


def test_parse_action_with_noise():
    d = parse_action('好的，我决定 {"action": "done", "result": "完成了"} 就这样')
    assert d["action"] == "done"


def test_parse_action_unknown_rejected():
    with pytest.raises(ValueError, match="未知动作"):
        parse_action('{"action": "delete_everything"}')


def test_parse_action_invalid():
    with pytest.raises(ValueError):
        parse_action("完全没有 JSON")


def test_parse_action_multiple_json_objects():
    """VLM 输出多个 JSON 时取第一个（贪婪正则会把多个对象抓成一团导致解析失败）。"""
    raw = '{"thought": "先分析", "x": 1}\n{"thought": "点8", "action": "click_id", "id": 33}'
    d = parse_action(raw)
    # 第一个是思考对象没有 action → 应报未知动作而非解析崩溃？
    # 设计决策：取第一个含 action 字段的对象
    assert d["action"] == "click_id" and d["id"] == 33


def test_parse_action_nested_braces_in_string():
    raw = '{"thought": "大括号 { 测试", "action": "done", "result": "ok"}'
    assert parse_action(raw)["action"] == "done"


# ---- 循环执行 ----

class ScriptProvider(VisionProvider):
    """按脚本依次返回决策的假 VLM。"""

    def __init__(self, script: list[str]):
        self.script = script
        self.calls = 0

    def pick_element(self, image_base64, legend, goal):
        return -1

    def ask_about_screen(self, image_base64, question):
        text = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return text


def _vlm_settings():
    return Settings(vision_provider="custom", vision_base_url="http://x/v1",
                    vision_api_key="k", vision_model="m")


def _loop_tools(provider) -> ScreenUse:
    return ScreenUse(dry_run=True, settings=_vlm_settings(), provider=provider)


FAKE_SHOT = {
    "image_base64": "ZmFrZQ==",
    "scale": 2.0,
    "width": 640,
    "height": 400,
    "elements": [el.to_dict() for el in FAKE_ELEMENTS],
}


def test_loop_click_then_done():
    provider = ScriptProvider([
        '{"thought": "点击按钮1", "action": "click_id", "id": 1}',
        '{"thought": "已完成", "action": "done", "result": "点完了"}',
    ])
    tools = _loop_tools(provider)
    tools.list_ui_elements()  # 填充 click_element_id 的缓存
    with patch.object(ScreenUse, "screenshot", return_value=FAKE_SHOT), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=5).run("点一下1号按钮")

    assert result.done is True
    assert result.result == "点完了"
    assert len(result.steps) == 2
    assert result.steps[0].action == "click_id"
    assert provider.calls == 2


def test_loop_coordinate_scaling():
    """VLM 返回的是缩放图坐标，需乘 scale 换算回真实屏幕。"""
    provider = ScriptProvider([
        '{"action": "click", "x": 100, "y": 50}',
        '{"action": "done", "result": "ok"}',
    ])
    tools = _loop_tools(provider)
    with patch.object(ScreenUse, "screenshot", return_value=FAKE_SHOT), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools).run("点坐标")
    # scale=2.0 → 真实坐标 (200, 100)
    assert result.steps[0].detail == "点击 (200, 100)"


def test_loop_max_steps():
    provider = ScriptProvider(['{"action": "wait", "seconds": 0}'])
    tools = _loop_tools(provider)
    with patch.object(ScreenUse, "screenshot", return_value=FAKE_SHOT), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=3).run("永远完不成的任务")
    assert result.done is False
    assert len(result.steps) == 3


def test_loop_action_failure_continues():
    """动作失败不中断，记入历史让 VLM 下一轮纠偏。"""
    provider = ScriptProvider([
        '{"action": "click_id", "id": 999}',   # 不存在的元素 → 失败
        '{"action": "done", "result": "放弃"}',
    ])
    tools = _loop_tools(provider)
    with patch.object(ScreenUse, "screenshot", return_value=FAKE_SHOT), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=5).run("测试容错")
    assert result.done is True
    assert result.steps[0].ok is False
    assert "不存在" in result.steps[0].detail


def test_loop_think_error_continues():
    """VLM 连续返回垃圾/空响应时，记为 think_error 步骤继续循环而不是崩溃。"""
    provider = ScriptProvider(['垃圾输出没有JSON'] * 10)
    tools = _loop_tools(provider)
    with patch.object(ScreenUse, "screenshot", return_value=FAKE_SHOT), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=2).run("测试决策失败")
    assert result.done is False
    assert all(s.action == "think_error" and not s.ok for s in result.steps)
    assert len(result.steps) == 2
