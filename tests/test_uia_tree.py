"""UIA 控件树测试：UIElement 数据结构与过滤逻辑（mock 控件对象）。"""

from screen_use.perception import uia_tree
from screen_use.perception.uia_tree import (
    ElementList,
    UIElement,
    _collect_from,
    _is_valid,
    get_element_text,
    set_element_text,
)

import pytest


class _Rect:
    def __init__(self, left, top, right, bottom):
        self.left, self.top, self.right, self.bottom = left, top, right, bottom

    def width(self):
        return self.right - self.left

    def height(self):
        return self.bottom - self.top


class _Ctrl:
    def __init__(self, offscreen=False, rect=(0, 0, 100, 50)):
        self.IsOffscreen = offscreen
        self.BoundingRectangle = _Rect(*rect)


def test_element_center_and_area():
    el = UIElement(id=1, name="确定", control_type="ButtonControl", bbox=(10, 20, 110, 60))
    assert el.center == (60, 40)
    assert el.area == 100 * 40


def test_element_to_dict():
    el = UIElement(id=3, name="输入框", control_type="EditControl", bbox=(0, 0, 50, 20))
    d = el.to_dict()
    assert d["id"] == 3
    assert d["bbox"] == [0, 0, 50, 20]
    assert d["center"] == [25, 10]


def test_is_valid_filters_offscreen():
    assert not _is_valid(_Ctrl(offscreen=True))


def test_is_valid_filters_zero_size():
    assert not _is_valid(_Ctrl(rect=(10, 10, 10, 50)))   # width=0
    assert not _is_valid(_Ctrl(rect=(10, 10, 60, 10)))   # height=0


def test_is_valid_accepts_normal():
    assert _is_valid(_Ctrl())


def test_is_valid_tolerates_exceptions():
    class _Bad:
        @property
        def IsOffscreen(self):
            raise RuntimeError("COM 异常")

    assert not _is_valid(_Bad())


# ---- DFS 节点预算 ----

class _TreeCtrl:
    """模拟 UIA 控件树节点。"""

    def __init__(self, name="", ctype="ButtonControl", rect=(0, 0, 100, 50), children=None):
        self.Name = name
        self.ControlTypeName = ctype
        self.IsOffscreen = False
        self.BoundingRectangle = _Rect(*rect)
        self._children = children or []
        self.NativeWindowHandle = 1234

    def GetChildren(self):
        return self._children


def test_collect_visited_budget():
    """已访问节点到 MAX_VISITED 即停，不再遍历大树的其余部分。"""
    # 构造一棵宽大树：根节点挂 5000 个子按钮
    root = _TreeCtrl(ctype="WindowControl",
                     children=[_TreeCtrl(name=f"btn{i}") for i in range(5000)])
    visited = [0]
    elements = _collect_from(root, "窗口", 1234, 10**8, visited)
    assert visited[0] == uia_tree.MAX_VISITED
    assert len(elements) < 5000


def test_collect_caches_control():
    """收集时缓存 UIA 控件对象，供文本读写。"""
    child = _TreeCtrl(name="确定")
    root = _TreeCtrl(ctype="WindowControl", children=[child])
    elements = _collect_from(root, "窗口", 1234, 10**8, [0])
    assert len(elements) == 1
    assert elements[0].control is child


# ---- UIA 文本读写 ----

class _ValuePattern:
    def __init__(self, value=""):
        self.Value = value
        self.set_calls = []

    def SetValue(self, text):
        self.set_calls.append(text)
        self.Value = text


class _EditCtrl:
    def __init__(self, value=""):
        self._vp = _ValuePattern(value)

    def GetValuePattern(self):
        return self._vp


def test_get_element_text_value_pattern():
    el = UIElement(id=1, name="旧名", control_type="EditControl",
                   bbox=(0, 0, 10, 10), control=_EditCtrl("已输入内容"))
    assert get_element_text(el) == "已输入内容"


def test_get_element_text_fallback_to_name():
    el = UIElement(id=1, name="按钮名", control_type="ButtonControl",
                   bbox=(0, 0, 10, 10), control=None)
    assert get_element_text(el) == "按钮名"


def test_set_element_text_via_value_pattern():
    ctrl = _EditCtrl()
    el = UIElement(id=1, name="", control_type="EditControl",
                   bbox=(0, 0, 10, 10), control=ctrl)
    set_element_text(el, "你好世界")
    assert ctrl._vp.set_calls == ["你好世界"]
    assert get_element_text(el) == "你好世界"


def test_set_element_text_without_control_raises():
    el = UIElement(id=1, name="", control_type="EditControl", bbox=(0, 0, 10, 10))
    with pytest.raises(ValueError, match="没有缓存"):
        set_element_text(el, "x")


# ---- ElementList ----

def test_element_list_carries_foreground_title():
    els = ElementList([UIElement(id=1, name="a", control_type="ButtonControl",
                                 bbox=(0, 0, 1, 1))], foreground_title="记事本")
    assert els.foreground_title == "记事本"
    assert len(els) == 1 and els[0].name == "a"  # 当普通 list 用完全兼容


# ---- 配额前去重（2026-09-18 计算器 7/8/9 被配额截掉事故） ----

def test_dedup_by_identity():
    """(名称, 类型, 位置) 完全重复才去重；类型或位置不同则保留。"""
    from screen_use.perception.uia_tree import _dedup_by_identity

    els = [
        UIElement(id=1, name="File", control_type="ButtonControl", bbox=(0, 0, 10, 10)),
        UIElement(id=2, name="File", control_type="ButtonControl", bbox=(0, 0, 10, 10)),  # 完全重复
        UIElement(id=3, name="File", control_type="MenuItemControl", bbox=(0, 0, 10, 10)),  # 类型不同
        UIElement(id=4, name="File", control_type="ButtonControl", bbox=(0, 0, 20, 20)),  # 位置不同
    ]
    unique = _dedup_by_identity(els)
    assert [el.id for el in unique] == [1, 3, 4]


# ---- 观测成本：时间预算 + 窗口成本记忆（2026-09-19 接入 Jev 后观测成新瓶颈） ----

def test_collect_stops_at_deadline():
    """到时间预算即停：最多走一个检查周期，已收集的元素保留。"""
    import time as _time

    root = _TreeCtrl(ctype="WindowControl",
                     children=[_TreeCtrl(name=f"btn{i}") for i in range(5000)])
    visited = [0]
    elements = _collect_from(root, "慢窗口", 1234, 10**8, visited,
                             deadline=_time.perf_counter() - 1)  # 已过期
    assert len(elements) <= uia_tree.NODE_TIME_CHECK
    assert visited[0] <= uia_tree.NODE_TIME_CHECK + 1


def test_foreground_window_ignores_deadline():
    """前台窗口不限时（调用方传 deadline=None）——它是 agent 的工作面。"""
    root = _TreeCtrl(ctype="WindowControl",
                     children=[_TreeCtrl(name=f"btn{i}") for i in range(50)])
    elements = _collect_from(root, "前台", 1234, 10**8, [0], deadline=None)
    assert len(elements) == 50


class _Win:
    def __init__(self, name):
        self.Name = name


def test_window_cost_memory_ema(monkeypatch):
    """窗口成本记忆：没见过=0（排最前），之后按 EMA 更新，贵的排后面。"""
    monkeypatch.setattr(uia_tree, "_WINDOW_COST", {})
    assert uia_tree._window_cost(_Win("音乐播放器")) == 0.0
    uia_tree._remember_cost("音乐播放器", 800.0)
    assert uia_tree._window_cost(_Win("音乐播放器")) == 800.0
    uia_tree._remember_cost("音乐播放器", 100.0)          # EMA: 0.7*800 + 0.3*100
    assert uia_tree._window_cost(_Win("音乐播放器")) == pytest.approx(590.0)
    # 贵的排后面：便宜窗口先走，预算用尽时丢的是无关的后台应用
    wins = [_Win("音乐播放器"), _Win("记事本")]
    assert sorted(wins, key=uia_tree._window_cost)[0].Name == "记事本"


def test_window_cost_tolerates_bad_name(monkeypatch):
    class _BadName:
        @property
        def Name(self):
            raise RuntimeError("COM 异常")

    monkeypatch.setattr(uia_tree, "_WINDOW_COST", {})
    assert uia_tree._window_cost(_BadName()) == 0.0


# ---- 后台窗口元素表缓存（2026-09-19：观测成本攻坚） ----

class _FakeWin:
    """假顶层窗口（够 uia_tree 用：标题/句柄/矩形/子节点）。"""

    def __init__(self, name, handle, rect=(0, 0, 400, 300), ctype="WindowControl",
                 offscreen=False):
        self.Name = name
        self.NativeWindowHandle = handle
        self.BoundingRectangle = _Rect(*rect)
        self.ControlTypeName = ctype
        self.IsOffscreen = offscreen
        self._children: list = []

    def GetChildren(self):
        return self._children


class _FakeRoot:
    def __init__(self, windows):
        self._windows = windows

    def GetChildren(self):
        return self._windows


def _patch_world(monkeypatch, windows, foreground):
    """把 UIA 世界换成假窗口；_collect_from 记录每个被真正遍历的窗口。"""
    walked: list[str] = []

    def fake_collect(w, title, handle, screen_area, visited, max_visited=0, deadline=None):
        walked.append(title)
        return [UIElement(id=0, name=f"{title}-按钮", control_type="ButtonControl",
                          bbox=(10, 10, 60, 40), window_title=title,
                          window_handle=handle)]

    monkeypatch.setattr(uia_tree, "_WINDOW_CACHE", {})
    monkeypatch.setattr(uia_tree, "_WINDOW_COST", {})
    monkeypatch.setattr(uia_tree, "_collect_from", fake_collect)
    monkeypatch.setattr(uia_tree, "_screen_size", lambda: (1280, 800))
    monkeypatch.setattr("screen_use.perception.screen.ensure_dpi_aware", lambda: None)
    monkeypatch.setattr("uiautomation.GetRootControl", lambda: _FakeRoot(windows))
    monkeypatch.setattr("uiautomation.GetForegroundControl", lambda: foreground)
    return walked


def test_background_window_cache_hit(monkeypatch):
    """第二次枚举：前台重走，后台窗口命中缓存（省掉整棵遍历）。"""
    fg = _FakeWin("前台", 1)
    bg = _FakeWin("慢后台", 2, rect=(0, 0, 300, 200))
    walked = _patch_world(monkeypatch, [fg, bg], fg)

    first = uia_tree.list_ui_elements(budget_ms=0)
    assert walked == ["前台", "慢后台"]
    assert first.timings["reused"] == []

    walked.clear()
    second = uia_tree.list_ui_elements(budget_ms=0)
    assert walked == ["前台"]                      # 后台没重走
    assert second.timings["reused"] == ["慢后台"]
    assert second.timings["cache_entries"] == 2
    # 缓存命中的窗口元素仍在结果里（不缩水）
    assert {e.name for e in second} == {"前台-按钮", "慢后台-按钮"}


def test_background_window_cache_disabled(monkeypatch):
    """cache_ttl_s=0 → 每次全走（老行为）。"""
    fg = _FakeWin("前台", 1)
    bg = _FakeWin("慢后台", 2)
    walked = _patch_world(monkeypatch, [fg, bg], fg)
    uia_tree.list_ui_elements(budget_ms=0, cache_ttl_s=0)
    walked.clear()
    uia_tree.list_ui_elements(budget_ms=0, cache_ttl_s=0)
    assert walked == ["前台", "慢后台"]


def test_background_window_cache_expires(monkeypatch):
    """TTL 过期 → 重走（用负 TTL 模拟已过期）。"""
    fg = _FakeWin("前台", 1)
    bg = _FakeWin("慢后台", 2)
    walked = _patch_world(monkeypatch, [fg, bg], fg)
    uia_tree.list_ui_elements(budget_ms=0)
    walked.clear()
    uia_tree.list_ui_elements(budget_ms=0, cache_ttl_s=-1)
    assert walked == ["前台", "慢后台"]


def test_background_window_cache_invalidated_on_rect_change(monkeypatch):
    """窗口矩形变了（移动/缩放）→ 该窗口重走，不复用旧坐标。"""
    fg = _FakeWin("前台", 1)
    bg = _FakeWin("慢后台", 2, rect=(0, 0, 300, 200))
    walked = _patch_world(monkeypatch, [fg, bg], fg)
    uia_tree.list_ui_elements(budget_ms=0)
    walked.clear()
    bg.BoundingRectangle = _Rect(500, 500, 800, 700)   # 窗口动了
    uia_tree.list_ui_elements(budget_ms=0)
    assert walked == ["前台", "慢后台"]


def test_background_window_cache_force_full(monkeypatch):
    """force_full=True → 忽略缓存全走（UI 大变动后要新鲜数据）。"""
    fg = _FakeWin("前台", 1)
    bg = _FakeWin("慢后台", 2)
    walked = _patch_world(monkeypatch, [fg, bg], fg)
    uia_tree.list_ui_elements(budget_ms=0)
    walked.clear()
    uia_tree.list_ui_elements(budget_ms=0, force_full=True)
    assert walked == ["前台", "慢后台"]


def test_window_cache_pruned_when_window_closes(monkeypatch):
    """窗口关掉后缓存条目要清掉（防句柄复用张冠李戴）。"""
    fg = _FakeWin("前台", 1)
    bg = _FakeWin("慢后台", 2)
    walked = _patch_world(monkeypatch, [fg, bg], fg)
    uia_tree.list_ui_elements(budget_ms=0)
    assert uia_tree.cache_stats()["entries"] == 2
    monkeypatch.setattr("uiautomation.GetRootControl", lambda: _FakeRoot([fg]))
    uia_tree.list_ui_elements(budget_ms=0)
    assert uia_tree.cache_stats()["entries"] == 1


def test_invalidate_window_cache():
    uia_tree._WINDOW_CACHE[42] = {"title": "x", "bbox": None, "elements": [], "ts": 0}
    uia_tree.invalidate_window_cache([42])
    assert uia_tree.cache_stats()["entries"] == 0
    uia_tree._WINDOW_CACHE[43] = {"title": "y", "bbox": None, "elements": [], "ts": 0}
    uia_tree.invalidate_window_cache()
    assert uia_tree.cache_stats()["entries"] == 0


# ---- 元素实时校验（缓存/点击前的安全阀） ----

def _el_with_rect(bbox, live_rect=None, raises=False):
    class _Ctl:
        @property
        def BoundingRectangle(self):
            if raises:
                raise RuntimeError("元素已失效")
            return _Rect(*live_rect)

    return UIElement(id=1, name="按钮", control_type="ButtonControl", bbox=bbox,
                     control=_Ctl())


def test_element_still_valid_same_place():
    assert uia_tree.element_still_valid(_el_with_rect((10, 10, 60, 40), (10, 10, 60, 40)))


def test_element_still_valid_tolerates_small_shift():
    """±4px 内的抖动不算移位（Qt 布局/边框取整会有一两像素差）。"""
    assert uia_tree.element_still_valid(_el_with_rect((10, 10, 60, 40), (13, 8, 62, 43)))


def test_element_still_valid_detects_move():
    assert not uia_tree.element_still_valid(_el_with_rect((10, 10, 60, 40), (200, 200, 250, 230)))


def test_element_still_valid_detects_dead_element():
    assert not uia_tree.element_still_valid(_el_with_rect((10, 10, 60, 40), raises=True))


def test_element_still_valid_detects_zero_size():
    assert not uia_tree.element_still_valid(_el_with_rect((10, 10, 60, 40), (10, 10, 10, 40)))


def test_element_still_valid_without_control_is_ok():
    """没有 control 对象（离线合成/测试）时不阻断。"""
    el = UIElement(id=1, name="x", control_type="ButtonControl", bbox=(0, 0, 10, 10))
    assert uia_tree.element_still_valid(el)
