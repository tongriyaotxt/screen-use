"""UIA 控件树测试：UIElement 数据结构与过滤逻辑（mock 控件对象）。"""

from screen_use.perception.uia_tree import UIElement, _is_valid


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
