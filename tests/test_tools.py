"""ScreenUse 门面测试：策略链、缓存、降级（全 mock，不依赖真实桌面/VLM）。"""

from unittest.mock import patch

import pytest

from screen_use.config import Settings
from screen_use.perception.uia_tree import UIElement
from screen_use.providers.base import VisionProvider
from screen_use.tools import ScreenUse

FAKE_ELEMENTS = [
    UIElement(id=1, name="确定", control_type="ButtonControl", bbox=(10, 10, 60, 40)),
    UIElement(id=2, name="取消", control_type="ButtonControl", bbox=(70, 10, 120, 40)),
    UIElement(id=3, name="文件名输入框", control_type="EditControl", bbox=(10, 50, 300, 80)),
]


def _no_vlm_settings() -> Settings:
    return Settings(vision_provider="openai", vision_api_key="", vision_model="")


class FakeProvider(VisionProvider):
    def __init__(self, pick=-1):
        self._pick = pick
        self.pick_calls = 0

    def pick_element(self, image_base64, legend, goal):
        self.pick_calls += 1
        return self._pick

    def ask_about_screen(self, image_base64, question):
        return f"回答: {question}"


def _make_tools(provider=None, settings=None) -> ScreenUse:
    tools = ScreenUse(dry_run=True, settings=settings or _no_vlm_settings(), provider=provider)
    return tools


@pytest.fixture(autouse=True)
def mock_uia(monkeypatch):
    monkeypatch.setattr(
        "screen_use.tools.uia_tree.list_ui_elements",
        lambda **kwargs: list(FAKE_ELEMENTS),
    )


# ---- UIA 文本匹配（策略链第 1 级） ----

def test_find_element_uia_exact():
    tools = _make_tools()
    result = tools.find_element("确定")
    assert result["found"] and result["method"] == "uia_exact"
    assert result["element"]["id"] == 1


def test_find_element_uia_fuzzy():
    tools = _make_tools()
    result = tools.find_element("输入框")  # 模糊匹配 "文件名输入框"
    assert result["found"] and result["method"] == "uia_fuzzy"
    assert result["element"]["id"] == 3


def test_find_element_uia_miss_without_vlm():
    tools = _make_tools()
    result = tools.find_element("不存在的按钮")
    assert not result["found"]
    assert "VLM 未配置" in result["error"]


# ---- VLM 层级（策略链第 2 级） ----

def _vlm_settings() -> Settings:
    return Settings(vision_provider="custom", vision_base_url="http://x/v1",
                    vision_api_key="k", vision_model="m")


def test_find_element_falls_back_to_vlm():
    provider = FakeProvider(pick=2)
    tools = _make_tools(provider=provider, settings=_vlm_settings())
    with patch("screen_use.tools.screen.screenshot") as mock_shot:
        from PIL import Image
        mock_shot.return_value = Image.new("RGB", (800, 600), "white")
        result = tools.find_element("某个图标")
    assert result["found"] and result["method"] == "vlm"
    assert result["element"]["id"] == 2
    assert provider.pick_calls == 1


def test_find_element_vlm_not_found():
    provider = FakeProvider(pick=-1)
    tools = _make_tools(provider=provider, settings=_vlm_settings())
    with patch("screen_use.tools.screen.screenshot") as mock_shot:
        from PIL import Image
        mock_shot.return_value = Image.new("RGB", (800, 600), "white")
        result = tools.find_element("某个图标")
    assert not result["found"] and result["method"] == "vlm"


# ---- click_element / click_element_id ----

def test_click_element_uia_path():
    tools = _make_tools()
    result = tools.click_element("取消")
    assert result["found"] and result["action"]["ok"]
    assert result["action"]["detail"].startswith("(95, 25)")  # 取消按钮中心点


def test_click_element_confirm_rejected():
    tools = _make_tools()
    tools.confirm_callback = lambda desc: False
    with pytest.raises(PermissionError):
        tools.click_element("确定")


def test_click_element_id_uses_cache():
    tools = _make_tools()
    tools.list_ui_elements()  # 填充缓存
    result = tools.click_element_id(3)
    assert result["ok"] and result["element"]["name"] == "文件名输入框"


def test_click_element_id_invalid():
    tools = _make_tools()
    tools.list_ui_elements()
    with pytest.raises(ValueError, match="不存在"):
        tools.click_element_id(999)


# ---- 缓存过期安全阀：点击前 bbox 校验 + 刷新重解析（2026-09-19） ----

class _CtlWithRect:
    """只有 BoundingRectangle 的假控件（喂给 element_still_valid）。"""

    def __init__(self, rect):
        self._rect = _FakeRect(*rect)

    @property
    def BoundingRectangle(self):
        return self._rect


def test_click_element_id_refreshes_stale_element(monkeypatch):
    """缓存里那个元素已经移位 → 先刷新再点新坐标，绝不点旧位置。"""
    tools = _make_tools()
    tools._elements = [UIElement(id=1, name="确定", control_type="ButtonControl",
                                 bbox=(10, 10, 60, 40),
                                 control=_CtlWithRect((310, 210, 360, 240)))]
    fresh = [UIElement(id=7, name="确定", control_type="ButtonControl",
                       bbox=(310, 210, 360, 240))]
    calls = []

    def fake_refresh(self, budget_ms=None, force_full=False):
        calls.append(force_full)
        self._elements = fresh
        return fresh

    monkeypatch.setattr(ScreenUse, "list_ui_elements_raw", fake_refresh)
    result = tools.click_element_id(1)
    assert calls == [True]                       # 走了强制刷新（不吃缓存）
    assert result["element"]["id"] == 7          # 点的是刷新后的元素
    assert result["element"]["center"] == [335, 225]


def test_click_element_id_raises_when_element_vanished(monkeypatch):
    """刷新后也找不到该元素 → 明确报错（交给上层重新观察），不猜着点。"""
    tools = _make_tools()
    tools._elements = [UIElement(id=1, name="已消失的按钮", control_type="ButtonControl",
                                 bbox=(10, 10, 60, 40),
                                 control=_CtlWithRect((310, 210, 360, 240)))]
    monkeypatch.setattr(ScreenUse, "list_ui_elements_raw",
                        lambda self, budget_ms=None, force_full=False: list(FAKE_ELEMENTS))
    with pytest.raises(ValueError, match="已不在界面上"):
        tools.click_element_id(1)


def test_click_element_id_skips_refresh_when_in_place():
    """元素还在原地 → 不刷新（省一次全量枚举）。"""
    tools = _make_tools()
    tools._elements = [UIElement(id=1, name="确定", control_type="ButtonControl",
                                 bbox=(10, 10, 60, 40),
                                 control=_CtlWithRect((10, 10, 60, 40)))]
    with patch.object(ScreenUse, "list_ui_elements_raw",
                      side_effect=AssertionError("不该刷新")):
        result = tools.click_element_id(1)
    assert result["ok"] is True


def test_list_ui_elements_passes_cache_ttl(monkeypatch):
    """配置里的缓存 TTL 透传给枚举（0=关缓存，必须真的传到）。"""
    seen = []

    def fake_list(**kwargs):
        seen.append(kwargs)
        return FAKE_ELEMENTS

    monkeypatch.setattr("screen_use.tools.uia_tree.list_ui_elements", fake_list)
    tools = _make_tools(settings=Settings(vision_provider="openai",
                                          uia_window_cache_ttl_s=1.5))
    tools.list_ui_elements()
    assert seen[0]["cache_ttl_s"] == 1.5
    assert seen[0]["force_full"] is False


# ---- read_screen 降级 ----

def test_read_screen_requires_vlm():
    tools = _make_tools()  # 无 VLM 配置
    with pytest.raises(RuntimeError, match="VLM 未配置"):
        tools.read_screen("屏幕上有什么？")


def test_read_screen_with_vlm():
    provider = FakeProvider()
    tools = _make_tools(provider=provider, settings=_vlm_settings())
    with patch("screen_use.tools.screen.screenshot") as mock_shot:
        from PIL import Image
        mock_shot.return_value = Image.new("RGB", (800, 600), "white")
        assert "回答" in tools.read_screen("有什么？")


# ---- 元学习：词汇映射（策略链第 0 级） ----

def test_find_element_learned_mapping():
    """第二次找同一描述时走 learned 路径，不再经过 UIA 匹配。"""
    tools = _make_tools()
    # 第一次：UIA 精确命中，同时学习映射
    r1 = tools.find_element("确定")
    assert r1["method"] == "uia_exact"
    assert tools.memory.recall_mapping("确定")["name"] == "确定"
    # 第二次：走 learned
    r2 = tools.find_element("确定")
    assert r2["method"] == "learned"
    assert r2["element"]["id"] == 1


def test_vlm_hit_learns_mapping():
    """VLM 定位成功后也会学习映射。"""
    provider = FakeProvider(pick=2)
    tools = _make_tools(provider=provider, settings=_vlm_settings())
    with patch("screen_use.tools.screen.screenshot") as mock_shot2:
        from PIL import Image
        mock_shot2.return_value = Image.new("RGB", (800, 600), "white")
        result = tools.find_element("叉叉按钮")
    assert result["method"] == "vlm"
    assert tools.memory.recall_mapping("叉叉按钮")["name"] == "取消"


# ---- list_ui_elements 返回结构（foreground_title） ----

def test_list_ui_elements_returns_foreground_title():
    tools = _make_tools()
    result = tools.list_ui_elements()
    assert "foreground_title" in result and "elements" in result
    assert len(result["elements"]) == 3


# ---- 观测成本：时间预算透传 + 空表兜底 + 元素表复用（2026-09-19） ----

def test_list_ui_elements_passes_budget(monkeypatch):
    """配置里的 uia_budget_ms 透传给枚举；预算不该被静默忽略。"""
    seen = []

    def fake_list(**kwargs):
        seen.append(kwargs.get("budget_ms"))
        return FAKE_ELEMENTS

    monkeypatch.setattr("screen_use.tools.uia_tree.list_ui_elements", fake_list)
    tools = _make_tools(settings=Settings(vision_provider="openai", uia_budget_ms=250))
    tools.list_ui_elements()
    assert seen == [250]


def test_list_ui_elements_falls_back_when_budget_empties_table(monkeypatch):
    """预算把元素表清空 → 不限时重来一次（宁可慢，不可瞎）。"""
    seen = []

    def fake_list(**kwargs):
        seen.append(kwargs.get("budget_ms"))
        return [] if kwargs.get("budget_ms") else FAKE_ELEMENTS

    monkeypatch.setattr("screen_use.tools.uia_tree.list_ui_elements", fake_list)
    tools = _make_tools(settings=Settings(vision_provider="openai", uia_budget_ms=100))
    result = tools.list_ui_elements()
    assert seen == [100, 0]
    assert len(result["elements"]) == 3


def test_screenshot_reuses_given_elements():
    """screenshot(elements=...) 复用调用方刚枚举的表，不再枚举一次。"""
    from PIL import Image

    tools = _make_tools()
    tools._foreground_title = "记事本"
    with patch.object(ScreenUse, "_enum_elements_in_thread",
                      side_effect=AssertionError("不该重复枚举")), \
         patch("screen_use.tools.screen.screenshot",
               return_value=Image.new("RGB", (800, 600), "white")):
        shot = tools.screenshot(annotate=True, elements=FAKE_ELEMENTS)
    assert [e["id"] for e in shot["elements"]] == [1, 2, 3]
    assert shot["foreground_title"] == "记事本"
    assert shot["image_base64"]


def test_screenshot_enumerates_when_not_given_elements():
    """不给 elements 时照旧自己枚举（老行为不变）。"""
    from PIL import Image

    tools = _make_tools()
    with patch("screen_use.tools.screen.screenshot",
               return_value=Image.new("RGB", (800, 600), "white")):
        shot = tools.screenshot(annotate=True)
    assert len(shot["elements"]) == 3


# ---- click_scaled ----

def test_click_scaled_converts_by_last_scale():
    tools = _make_tools()
    tools._last_scale = 2.0
    result = tools.click_scaled(100, 50)
    assert result["detail"].startswith("(200, 100)")


def test_click_scaled_default_scale_1():
    tools = _make_tools()
    result = tools.click_scaled(30, 40)
    assert result["detail"].startswith("(30, 40)")


# ---- UIA 文本读写（SDK 层） ----

class _FakeVP:
    def __init__(self, value=""):
        self.Value = value

    def SetValue(self, text):
        self.Value = text


class _FakeEdit:
    def __init__(self, value="", bbox=(0, 0, 100, 30)):
        self._vp = _FakeVP(value)
        self._rect = _FakeRect(*bbox)

    def GetValuePattern(self):
        return self._vp

    def get_BoundingRectangle(self):
        return self._rect

    @property
    def BoundingRectangle(self):
        return self._rect


class _FakeRect:
    def __init__(self, left, top, right, bottom):
        self.left, self.top, self.right, self.bottom = left, top, right, bottom

    def width(self):
        return self.right - self.left

    def height(self):
        return self.bottom - self.top


def _tools_with_edit(value="", bbox=(0, 0, 100, 30), control=None):
    tools = _make_tools()
    edit = UIElement(id=9, name="输入框", control_type="EditControl",
                     bbox=bbox, control=control if control is not None else _FakeEdit(value, bbox))
    tools._elements = [edit]
    return tools


def test_get_element_text():
    tools = _tools_with_edit("hello")
    assert tools.get_element_text(9) == {"ok": True, "text": "hello"}


def test_set_element_text():
    tools = _tools_with_edit()
    result = tools.set_element_text(9, "绕过禁粘贴")
    assert result["ok"] is True
    assert tools.get_element_text(9)["text"] == "绕过禁粘贴"


def test_set_element_text_invalid_id():
    tools = _tools_with_edit()
    with pytest.raises(ValueError, match="不存在"):
        tools.set_element_text(999, "x")


# ---- do_actions 批量动作 ----

FAKE_SHOT = {
    "image_base64": "ZmFrZQ==",
    "scale": 1.0,
    "width": 100,
    "height": 100,
    "quality": 80,
    "foreground_title": "",
    "elements": [],
}


def test_do_actions_sequence():
    tools = _make_tools()
    with patch.object(ScreenUse, "screenshot", return_value=dict(FAKE_SHOT)), \
         patch("screen_use.tools.time.sleep") as mock_sleep:
        result = tools.do_actions([
            {"action": "click", "x": 10, "y": 20},
            {"action": "type_text", "text": "abc"},
            {"action": "press", "key": "enter"},
        ], interval=0.3)
    assert [r["ok"] for r in result["results"]] == [True, True, True]
    assert mock_sleep.call_count == 2  # 每步间 sleep，最后一步后不 sleep
    assert result["screenshot"]["width"] == 100


def test_do_actions_stops_on_failure():
    tools = _make_tools()
    tools.list_ui_elements()  # 填充缓存
    with patch.object(ScreenUse, "screenshot", return_value=dict(FAKE_SHOT)), \
         patch("screen_use.tools.time.sleep"):
        result = tools.do_actions([
            {"action": "click_element_id", "element_id": 999},  # 不存在 → 失败
            {"action": "press", "key": "enter"},                # 不应执行
        ])
    assert len(result["results"]) == 1
    assert result["results"][0]["ok"] is False
    assert "不存在" in result["results"][0]["detail"]


def test_do_actions_unknown_action():
    tools = _make_tools()
    with patch.object(ScreenUse, "screenshot", return_value=dict(FAKE_SHOT)):
        result = tools.do_actions([{"action": "explode"}])
    assert result["results"][0]["ok"] is False
    assert "不支持的动作" in result["results"][0]["detail"]


# ---- screenshot(annotate=True)：UIA 枚举与截图并行 ----

def test_screenshot_annotate_parallel_returns_elements():
    """annotate=True 时元素枚举（线程内）与截图并行，结果包含元素列表和标注图。"""
    from PIL import Image

    tools = _make_tools()
    with patch("screen_use.tools.screen.screenshot",
               return_value=Image.new("RGB", (800, 600), "white")):
        result = tools.screenshot(annotate=True)
    assert len(result["elements"]) == 3  # 来自 mock_uia 的 FAKE_ELEMENTS
    assert result["image_base64"]


def test_screenshot_annotate_falls_back_when_thread_fails(monkeypatch):
    """线程内枚举失败（如 COM 环境异常）时，主线程顺序枚举兜底。"""
    from PIL import Image

    def boom(self):
        raise RuntimeError("COM 爆炸")

    monkeypatch.setattr(ScreenUse, "_enum_elements_in_thread", boom)
    tools = _make_tools()
    with patch("screen_use.tools.screen.screenshot",
               return_value=Image.new("RGB", (800, 600), "white")):
        result = tools.screenshot(annotate=True)
    assert len(result["elements"]) == 3  # 兜底路径仍拿到元素（mock_uia）


# ---- 阶段 A：三环协议（get_capabilities + EscalationResult） ----

LOOP_SHOT = {
    "image_base64": "ZmFrZQ==",
    "scale": 2.0,
    "width": 640,
    "height": 400,
    "elements": [el.to_dict() for el in FAKE_ELEMENTS],
}


def _vlm_tools(provider) -> ScreenUse:
    settings = Settings(vision_provider="custom", vision_base_url="http://x/v1",
                        vision_api_key="k", vision_model="m")
    return ScreenUse(dry_run=True, settings=settings, provider=provider)


class _ScriptProvider(VisionProvider):
    def __init__(self, script):
        self.script = script
        self.calls = 0

    def pick_element(self, image_base64, legend, goal):
        return -1

    def ask_about_screen(self, image_base64, question):
        text = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return text


def test_get_capabilities_manifest():
    caps = _make_tools().get_capabilities()
    assert set(caps["rings"]) == {"skill", "task", "atomic"}
    # 诚实标注：skill 未实现、task 当前引擎是 vlm
    assert caps["rings"]["skill"]["available"] is False
    assert "vlm" in caps["rings"]["task"]["engine"]
    # 说明书必须有 cannot（写清不能做什么比能做什么更重要）
    assert caps["rings"]["task"]["cannot"]
    assert "escalation" in caps and "routing_hint" in caps


def test_run_task_done_returns_status_done():
    provider = _ScriptProvider(['{"action": "done", "result": "ok"}'])
    tools = _vlm_tools(provider)
    with patch.object(ScreenUse, "screenshot", return_value=LOOP_SHOT), \
         patch("screen_use.agent.time.sleep"):
        result = tools.run_task("测试完成", max_steps=3)
    assert result["status"] == "done"
    assert result["done"] is True and result["result"] == "ok"


def test_run_task_max_steps_returns_escalation_with_map():
    """任务未完成 → EscalationResult：原因分类 + 进度 + 当前观测快照（带地图退）。"""
    provider = _ScriptProvider(['{"action": "wait", "seconds": 0}'])
    tools = _vlm_tools(provider)
    with patch.object(ScreenUse, "screenshot", return_value=LOOP_SHOT), \
         patch("screen_use.agent.time.sleep"):
        result = tools.run_task("永远完不成", max_steps=2)

    assert result["status"] == "escalated"
    assert result["from_ring"] == "task"
    assert result["reason"]["class"] == "max_steps_reached"
    assert len(result["progress"]["steps_done"]) == 2
    assert result["progress"]["remaining_goal"] == "永远完不成"
    # 观测快照：宿主可直接接管，无需重新截图枚举
    assert result["observation"]["elements"][0]["name"] == "确定"
    assert result["suggestion"]


def test_run_task_think_error_escalation_class():
    """VLM 连续返回垃圾 → reason.class = think_error，建议检查 VLM 配置。"""
    provider = _ScriptProvider(["垃圾输出"] * 10)
    tools = _vlm_tools(provider)
    with patch.object(ScreenUse, "screenshot", return_value=LOOP_SHOT), \
         patch("screen_use.agent.time.sleep"):
        result = tools.run_task("测试决策失败", max_steps=3)
    assert result["status"] == "escalated"
    assert result["reason"]["class"] == "think_error"
    assert "VLM" in result["suggestion"]
