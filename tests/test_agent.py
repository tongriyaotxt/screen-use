"""视觉行为循环测试：parse_action + 循环执行（mock provider，dry_run）。"""

from unittest.mock import patch

import pytest

from screen_use.agent import VisualLoop, parse_action, parse_actions
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


# ---- parse_actions（batch 解析，向后兼容单动作） ----

def test_parse_actions_single():
    actions = parse_actions('{"thought": "点1", "action": "click_id", "id": 1}')
    assert len(actions) == 1 and actions[0]["action"] == "click_id"


def test_parse_actions_batch():
    raw = '{"thought": "连续操作", "batch": [' \
          '{"action": "click_id", "id": 3}, ' \
          '{"action": "type", "text": "你好"}, ' \
          '{"action": "press", "key": "enter"}]}'
    actions = parse_actions(raw)
    assert [a["action"] for a in actions] == ["click_id", "type", "press"]
    assert actions[0]["thought"] == "连续操作"  # batch 的 thought 带给第一个动作


def test_parse_actions_batch_filters_invalid():
    """batch 中混有未知动作时过滤掉，保留合法的。"""
    raw = '{"batch": [{"action": "explode"}, {"action": "wait", "seconds": 1}]}'
    actions = parse_actions(raw)
    assert [a["action"] for a in actions] == ["wait"]


def test_parse_actions_invalid_raises():
    with pytest.raises(ValueError):
        parse_actions("完全没有 JSON")


# ---- batch 循环执行 ----

def test_loop_batch_executes_multiple_actions():
    """模型一轮返回 batch 时，一轮内依次执行多个动作。"""
    provider = ScriptProvider([
        '{"thought": "连续操作", "batch": ['
        '{"action": "click_id", "id": 1}, '
        '{"action": "type", "text": "hi"}, '
        '{"action": "press", "key": "enter"}]}',
        '{"thought": "已完成", "action": "done", "result": "搞定"}',
    ])
    tools = _loop_tools(provider)
    tools.list_ui_elements()  # 填充 click_element_id 的缓存
    with patch.object(ScreenUse, "screenshot", return_value=FAKE_SHOT), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=5).run("批量操作")

    assert result.done is True
    assert [s.action for s in result.steps] == ["click_id", "type", "press", "done"]
    assert provider.calls == 2  # batch 一轮一次 VLM 调用


def test_loop_batch_stops_on_failure():
    """batch 中某步失败即中止本轮剩余动作，下轮重新观察。"""
    provider = ScriptProvider([
        '{"batch": [{"action": "click_id", "id": 999}, {"action": "press", "key": "enter"}]}',
        '{"action": "done", "result": "放弃"}',
    ])
    tools = _loop_tools(provider)
    with patch.object(ScreenUse, "screenshot", return_value=FAKE_SHOT), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=5).run("容错")
    assert result.done is True
    assert [s.action for s in result.steps] == ["click_id", "done"]  # press 未执行
    assert result.steps[0].ok is False


# ---- _think 纠正式重试 ----

class FeedbackProvider(VisionProvider):
    """第一次返回垃圾，第二次返回合法 JSON；记录收到的 prompt。"""

    def __init__(self):
        self.prompts: list[str] = []

    def pick_element(self, image_base64, legend, goal):
        return -1

    def ask_about_screen(self, image_base64, question):
        self.prompts.append(question)
        if len(self.prompts) == 1:
            return "垃圾输出没有JSON"
        return '{"action": "done", "result": "ok"}'


def test_think_retry_with_feedback():
    """解析失败重试时把错误原因反馈给模型（纠正式重试），且只重试 1 次即成功。"""
    provider = FeedbackProvider()
    tools = _loop_tools(provider)
    with patch.object(ScreenUse, "screenshot", return_value=FAKE_SHOT), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=3).run("测试纠错重试")
    assert result.done is True
    assert len(provider.prompts) == 2
    assert "纠错" in provider.prompts[1]  # 第二次调用带了解析错误反馈


# ---- _reflect 复用 observe 截图 ----

class RecordingProvider(VisionProvider):
    """按脚本返回，同时记录每次调用的图片和问题。"""

    def __init__(self, script: list[str]):
        self.script = script
        self.calls = 0
        self.images: list[str] = []
        self.questions: list[str] = []

    def pick_element(self, image_base64, legend, goal):
        return -1

    def ask_about_screen(self, image_base64, question):
        self.images.append(image_base64)
        self.questions.append(question)
        text = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return text


def test_reflect_reuses_observe_screenshot():
    """反思时复用本轮 observe 的截图（不重复截图），且反思结论注入下一轮 prompt。"""
    provider = RecordingProvider([
        '{"action": "click_id", "id": 1}',
        '{"action": "click_id", "id": 1}',               # 屏幕不变 → NO_EFFECT → 触发反思
        '{"analysis": "点不动", "strategy": "换双击"}',  # 反思响应
        '{"action": "done", "result": "ok"}',
    ])
    tools = _loop_tools(provider)
    tools.list_ui_elements()  # 填充 click_element_id 的缓存
    with patch.object(ScreenUse, "screenshot", return_value=FAKE_SHOT), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=5).run("测试反思复用截图")

    assert result.done is True
    reflect_idx = next(i for i, q in enumerate(provider.questions) if "内省反思" in q)
    # 反思用的是 observe 的截图（FAKE_SHOT 的 base64），不是重新截的
    assert provider.images[reflect_idx] == FAKE_SHOT["image_base64"]
    # 反思结论注入了后续 think 的 prompt
    later_prompts = provider.questions[reflect_idx + 1:]
    assert any("点不动" in q for q in later_prompts)


# ---- vlm_max_size ----

def test_loop_uses_vlm_max_size():
    """observe 截图用 vlm_max_size（发给 VLM 的图更小、TTFT 更低），而非 screenshot_max_size。"""
    provider = ScriptProvider(['{"action": "done", "result": "ok"}'])
    settings = Settings(vision_provider="custom", vision_base_url="http://x/v1",
                        vision_api_key="k", vision_model="m", vlm_max_size=640)
    tools = ScreenUse(dry_run=True, settings=settings, provider=provider)
    captured = {}

    def fake_screenshot(self, annotate=False, max_size=None, quality=None,
                        elements=None):
        captured["max_size"] = max_size
        captured["elements"] = elements
        return dict(FAKE_SHOT)

    with patch.object(ScreenUse, "screenshot", fake_screenshot), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=2).run("测试 vlm_max_size")
    assert result.done is True
    assert captured["max_size"] == 640


def test_vlm_max_size_default():
    """vlm_max_size 默认 896（不读 .env，隔离用户本机配置）。"""
    assert Settings(_env_file=None).vlm_max_size == 896


# ---- 结构化轨迹采集（desktop-jev 阶段 0） ----

import json as _json


def test_loop_records_full_trace(tmp_path):
    """run 每步落盘一条结构化轨迹（含元素表与决策），done 步也记录。"""
    from screen_use.memory import ExperienceStore

    provider = ScriptProvider([
        '{"thought": "点击按钮1", "action": "click_id", "id": 1}',
        '{"thought": "已完成", "action": "done", "result": "点完了"}',
    ])
    tools = _loop_tools(provider)
    tools.list_ui_elements()  # 填充 click_element_id 的缓存
    memory = ExperienceStore(tmp_path / "mem")
    with patch.object(ScreenUse, "screenshot", return_value=FAKE_SHOT), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=5, memory=memory).run("点一下1号按钮")
    assert result.done is True

    files = list((tmp_path / "mem" / "traces_full").glob("*.jsonl"))
    assert len(files) == 1
    steps = [_json.loads(l) for l in files[0].read_text(encoding="utf-8").splitlines()]
    assert [s["action"] for s in steps] == ["click_id", "done"]
    assert steps[0]["elements"] == FAKE_SHOT["elements"]
    assert steps[0]["goal"] == "点一下1号按钮"
    assert steps[0]["args"] == {"id": 1} and steps[0]["ok"] is True


def test_loop_record_full_trace_disabled(tmp_path):
    """record_full_trace=False 时不产生 traces_full 目录。"""
    from screen_use.memory import ExperienceStore

    provider = ScriptProvider(['{"action": "done", "result": "ok"}'])
    settings = _vlm_settings()
    settings.record_full_trace = False
    tools = ScreenUse(dry_run=True, settings=settings, provider=provider)
    memory = ExperienceStore(tmp_path / "mem")
    with patch.object(ScreenUse, "screenshot", return_value=FAKE_SHOT), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=2, memory=memory).run("关闭采集")
    assert result.done is True
    assert not (tmp_path / "mem" / "traces_full").exists()


def test_record_full_trace_default():
    """record_full_trace 默认开启（不读 .env，隔离用户本机配置）。"""
    assert Settings(_env_file=None).record_full_trace is True


# ---- Jev 快路径（desktop-jev 阶段 4 接入，cascade protocol 中环） ----

from screen_use.perception.uia_tree import ElementList  # noqa: E402


class FakeJev:
    """脚本化 Jev 引擎（不依赖 torch）：记录每次调用收到的观测与历史。"""

    def __init__(self, decisions: list[dict], latency_ms: float = 7.0):
        self.decisions = decisions
        self.latency_ms = latency_ms
        self.calls: list[dict] = []

    def decide(self, goal, foreground, elements, history):
        self.calls.append({"goal": goal, "foreground": foreground,
                           "elements": elements, "history": list(history)})
        d = dict(self.decisions[min(len(self.calls) - 1, len(self.decisions) - 1)])
        d.setdefault("confidence", 0.9)
        d.setdefault("escalate", d["confidence"] < 0.5)
        d["latency_ms"] = self.latency_ms
        return d


class BoomJev:
    """推理直接抛异常（模拟 CUDA OOM / 引擎未就绪）。"""

    def decide(self, *args, **kwargs):
        raise RuntimeError("CUDA out of memory")


@pytest.fixture
def mock_uia(monkeypatch):
    """快路径观测走 UIA 元素表：给它一个带前台标题的元素表。"""
    monkeypatch.setattr(
        "screen_use.tools.uia_tree.list_ui_elements",
        lambda **kwargs: ElementList(FAKE_ELEMENTS, foreground_title="MockUI-1"),
    )


def _no_screenshot(*_args, **_kwargs):
    raise AssertionError("Jev 快路径不应截图")


def test_jev_fast_path_skips_vlm_and_screenshot(mock_uia):
    """快路径：Jev 直接决策 → 零 VLM 调用、零截图，全程 ring=jev。"""
    provider = ScriptProvider(['{"action": "done", "result": "不该被调用"}'])
    tools = _loop_tools(provider)
    jev = FakeJev([
        {"action": "CLICK", "element_id": 1, "confidence": 0.91},
        {"action": "DONE", "confidence": 0.95},
    ])
    with patch.object(ScreenUse, "screenshot", _no_screenshot), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=5, jev=jev).run("点一下1号按钮")

    assert result.done is True
    assert provider.calls == 0                      # 一次 VLM 都没花
    assert [s.ring for s in result.steps] == ["jev", "jev"]
    assert result.steps[0].action == "click_id" and result.steps[0].args["id"] == 1
    assert result.metrics["jev_steps"] == 2
    assert result.metrics["vlm_steps"] == 0 and result.metrics["vlm_calls"] == 0
    assert result.metrics["screenshot_calls"] == 0  # SoM 截图+编码全省
    assert result.metrics["ring"] == "jev"
    assert result.metrics["jev_ms"] > 0 and result.metrics["wall_ms"] > 0


def test_jev_observation_passed_to_engine(mock_uia):
    """观测（前台标题 + 元素表）与 history 按训练同构格式喂给 Jev。"""
    provider = ScriptProvider(["不该被调用"])
    tools = _loop_tools(provider)
    jev = FakeJev([
        {"action": "TYPE", "element_id": 1, "text": "北京", "confidence": 0.9},
        {"action": "DONE", "confidence": 0.9},
    ])
    with patch.object(ScreenUse, "screenshot", _no_screenshot), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=3, jev=jev).run("在城市输入框输入北京")

    assert result.done is True
    assert jev.calls[0]["goal"] == "在城市输入框输入北京"
    assert jev.calls[0]["foreground"] == "MockUI-1"
    assert jev.calls[0]["elements"][0]["id"] == 1
    assert jev.calls[0]["history"] == []
    assert result.steps[0].action == "type" and result.steps[0].args["text"] == "北京"
    # 第二步拿到的 history 必须与 SFT 训练时同一格式
    assert jev.calls[1]["history"] == ["[TYPE 北京 ✓]"]


def test_jev_low_confidence_escalates_to_vlm_with_hint(mock_uia):
    """低置信 → 走原 VLM 路径，并把 Jev 的判断作为提示带过去（带地图退）。"""
    provider = RecordingProvider(['{"action": "done", "result": "慢路径完成"}'])
    tools = _loop_tools(provider)
    jev = FakeJev([{"action": "CLICK", "element_id": 1, "confidence": 0.2}])
    with patch.object(ScreenUse, "screenshot", return_value=FAKE_SHOT), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=3, jev=jev).run("点一下1号按钮")

    assert result.done is True and result.result == "慢路径完成"
    assert provider.calls == 1
    assert result.metrics["jev_escalations"] == 1
    assert result.metrics["vlm_steps"] == 1 and result.metrics["vlm_calls"] == 1
    assert result.metrics["screenshot_calls"] == 1   # 升级才补 SoM 截图
    # 本步最终由 VLM 决策完成，所以整体 ring=vlm；但升级计数如实记着
    assert result.metrics["ring"] == "vlm"
    assert result.steps[0].ring == "vlm"
    assert "快路径提示" in provider.questions[0]
    assert "CLICK" in provider.questions[0] and "#1" in provider.questions[0]


def test_jev_hybrid_ring_when_both_rings_act(mock_uia):
    """两环都真的执行过动作 → ring=hybrid（北极星指标要能区分三种驾驶模式）。"""
    provider = RecordingProvider(['{"action": "done", "result": "慢路径收尾"}'])
    tools = _loop_tools(provider)
    jev = FakeJev([
        {"action": "CLICK", "element_id": 1, "confidence": 0.9},
        {"action": "CLICK", "element_id": 1, "confidence": 0.1},
    ])
    with patch.object(ScreenUse, "screenshot", return_value=FAKE_SHOT), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=5, jev=jev).run("点一下1号按钮")

    assert result.done is True
    assert [s.ring for s in result.steps] == ["jev", "vlm"]
    assert result.metrics["ring"] == "hybrid"
    assert result.metrics["jev_steps"] == 1 and result.metrics["vlm_steps"] == 1
    assert result.metrics["jev_escalations"] == 1


def test_jev_blocked_escalates_without_burning_vlm(mock_uia):
    """Jev 认输（BLOCKED）→ 不烧 VLM，直接带观测快照升级（§4 blocked_action）。"""
    provider = ScriptProvider(["不该被调用"])
    tools = _loop_tools(provider)
    jev = FakeJev([{"action": "BLOCKED", "confidence": 0.99}])
    with patch.object(ScreenUse, "screenshot", _no_screenshot), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=3, jev=jev).run("点击不存在的按钮")

    assert result.done is False
    assert result.escalated is True
    assert result.escalation_reason == "blocked_action"
    assert provider.calls == 0
    assert result.steps[-1].action == "blocked" and result.steps[-1].ring == "jev"
    assert result.metrics["ring"] == "jev"


def test_jev_error_degrades_to_vlm(mock_uia):
    """引擎异常只降级不中断：记 jev_errors，本步回落 VLM。"""
    provider = ScriptProvider(['{"action": "done", "result": "ok"}'])
    tools = _loop_tools(provider)
    with patch.object(ScreenUse, "screenshot", return_value=FAKE_SHOT), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=2, jev=BoomJev()).run("任务")

    assert result.done is True
    assert result.metrics["jev_errors"] == 1
    assert result.metrics["vlm_steps"] == 1 and result.metrics["jev_steps"] == 0
    assert result.metrics["ring"] == "vlm"


def test_run_task_with_jev_reports_engine_and_metrics(mock_uia):
    """run_task 的交接包要带 engine 与 metrics（宿主据此判断该不该再试快路径）。"""
    provider = ScriptProvider(["不该被调用"])
    tools = _loop_tools(provider)
    tools.jev = FakeJev([{"action": "BLOCKED", "confidence": 0.9}])
    with patch.object(ScreenUse, "screenshot", _no_screenshot), \
         patch("screen_use.agent.time.sleep"):
        out = tools.run_task("点击不存在的按钮", max_steps=3)

    assert out["status"] == "escalated"
    assert out["from_ring"] == "jev" and out["engine"] == "jev"
    assert out["reason"]["class"] == "blocked_action"
    assert out["metrics"]["jev_steps"] == 1
    assert out["observation"]["foreground_title"] == "MockUI-1"
    assert out["progress"]["remaining_goal"] == "点击不存在的按钮"


def test_pure_vlm_metrics_ring(mock_uia):
    """未接 Jev 时指标应如实标注 ring=vlm，且每步一次截图（对照组基线）。"""
    provider = ScriptProvider(['{"action": "done", "result": "ok"}'])
    tools = _loop_tools(provider)
    with patch.object(ScreenUse, "screenshot", return_value=FAKE_SHOT), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=2).run("任务")
    assert result.metrics["ring"] == "vlm"
    assert result.metrics["vlm_steps"] == 1 and result.metrics["screenshot_calls"] == 1
    assert result.metrics["jev_steps"] == 0


# ---- 观测成本：升级时复用刚枚举的元素表（2026-09-19） ----

def test_escalation_reuses_enumerated_elements(mock_uia):
    """升级那一步复用刚枚举的元素表，不再多花一次 UIA 枚举（实测 ~0.4–1.4s）。"""
    provider = ScriptProvider(['{"action": "done", "result": "慢路径完成"}'])
    tools = _loop_tools(provider)
    jev = FakeJev([{"action": "CLICK", "element_id": 1, "confidence": 0.2}])
    captured = {}

    def fake_screenshot(self, annotate=False, max_size=None, quality=None,
                        elements=None):
        captured["elements"] = elements
        return dict(FAKE_SHOT)

    with patch.object(ScreenUse, "screenshot", fake_screenshot), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=3, jev=jev).run("点一下1号按钮")

    assert result.done is True
    assert captured["elements"] and captured["elements"][0].id == 1  # 复用而非重枚举
    assert result.metrics["enum_reuse"] == 1


def test_fast_path_does_not_reuse_across_steps(mock_uia):
    """跨步不复用：每步都重新枚举（动作可能改变界面）。"""
    provider = ScriptProvider(["不该被调用"])
    tools = _loop_tools(provider)
    jev = FakeJev([
        {"action": "CLICK", "element_id": 1, "confidence": 0.9},
        {"action": "DONE", "confidence": 0.9},
    ])
    with patch.object(ScreenUse, "screenshot", _no_screenshot), \
         patch("screen_use.agent.time.sleep"):
        result = VisualLoop(tools, max_steps=3, jev=jev).run("任务")
    assert result.done is True
    assert len(jev.calls) == 2
    assert "enum_reuse" not in result.metrics
