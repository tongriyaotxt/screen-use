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
        "screen_use.tools.uia_tree.list_ui_elements", lambda: list(FAKE_ELEMENTS)
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
