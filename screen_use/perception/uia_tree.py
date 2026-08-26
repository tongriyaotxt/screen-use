"""UIA 控件树枚举：获取前台窗口的可交互元素。

坐标为物理像素（需先调用 screen.ensure_dpi_aware），与截图/点击坐标一致。
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any

# 可交互控件类型白名单（uiautomation 的 ControlType 名称）
INTERACTIVE_TYPES = {
    "ButtonControl",
    "EditControl",
    "ComboBoxControl",
    "CheckBoxControl",
    "RadioButtonControl",
    "MenuItemControl",
    "ListItemControl",
    "TreeItemControl",
    "TabItemControl",
    "HyperlinkControl",
    "SliderControl",
    "SpinnerControl",
    "SplitButtonControl",
    "DataItemControl",
    "DocumentControl",
}

MAX_ELEMENTS = 150
MAX_WINDOWS = 5
MAX_DEPTH = 12
MAX_VISITED = 3000  # DFS 已访问节点预算：防止大页面（浏览器/IDE）秒级遍历


@dataclass
class UIElement:
    """一个可交互 UI 元素。bbox = (left, top, right, bottom)，物理像素。"""

    id: int
    name: str
    control_type: str
    bbox: tuple[int, int, int, int]
    window_title: str = ""
    window_handle: int = 0
    extra: dict = field(default_factory=dict)
    control: Any = field(default=None, repr=False, compare=False)  # UIA 控件对象（供文本读写，不序列化）

    @property
    def center(self) -> tuple[int, int]:
        left, top, right, bottom = self.bbox
        return ((left + right) // 2, (top + bottom) // 2)

    @property
    def area(self) -> int:
        left, top, right, bottom = self.bbox
        return max(0, right - left) * max(0, bottom - top)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "control_type": self.control_type,
            "bbox": list(self.bbox),
            "center": list(self.center),
            "window_title": self.window_title,
        }


def get_element_text(el: UIElement) -> str:
    """读控件文本：优先 ValuePattern（Edit 的真实内容），退回控件 Name。"""
    ctrl = el.control
    if ctrl is not None:
        try:
            vp = ctrl.GetValuePattern()
            if vp is not None:
                return vp.Value or ""
        except Exception:
            pass
    return el.name


def set_element_text(el: UIElement, text: str) -> None:
    """写控件文本：走 UIA ValuePattern.SetValue，绕过禁粘贴的输入框。

    失败时抛 ValueError（控件缓存过期/不支持 ValuePattern），由上层决定回退策略。
    """
    ctrl = el.control
    if ctrl is None:
        raise ValueError(f"元素 #{el.id} 没有缓存的 UIA 控件，请重新 list_ui_elements")
    try:
        vp = ctrl.GetValuePattern()
        if vp is None:
            raise ValueError("不支持 ValuePattern")
        vp.SetValue(text)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"SetValue 失败（控件可能已失效）: {e}") from e


class ElementList(list):
    """list[UIElement] + 枚举时的前台窗口标题（当普通 list 用完全向后兼容）。"""

    def __init__(self, iterable=(), foreground_title: str = ""):
        super().__init__(iterable)
        self.foreground_title = foreground_title


def _is_valid(ctrl) -> bool:
    """过滤不可见 / 无尺寸 / 离屏的控件。"""
    try:
        if ctrl.IsOffscreen:
            return False
        rect = ctrl.BoundingRectangle
        if rect.width() <= 0 or rect.height() <= 0:
            return False
        # 排除过大的容器（几乎全屏的通常不是可点目标）
        return True
    except Exception:
        return False


def _safe_handle(ctrl) -> int:
    try:
        return ctrl.NativeWindowHandle
    except Exception:
        return -1


def _collect_from(root_ctrl, window_title: str, window_handle: int, screen_area: int,
                  visited: list[int]) -> list[UIElement]:
    """从某个窗口的控件树收集可交互元素（深度优先）。

    visited: 跨窗口共享的已访问节点计数（[0] 为计数器），到 MAX_VISITED 即停，
    防止浏览器/IDE 等大页面的控件树导致秒级遍历。
    """
    elements: list[UIElement] = []

    def walk(ctrl, depth: int) -> None:
        if visited[0] >= MAX_VISITED or depth > MAX_DEPTH:
            return
        visited[0] += 1
        try:
            ctype = ctrl.ControlTypeName
        except Exception:
            ctype = ""
        if ctype in INTERACTIVE_TYPES and _is_valid(ctrl):
            rect = ctrl.BoundingRectangle
            bbox = (rect.left, rect.top, rect.right, rect.bottom)
            area = (rect.right - rect.left) * (rect.bottom - rect.top)
            if area < screen_area * 0.9:  # 排除近全屏控件
                elements.append(
                    UIElement(
                        id=0,
                        name=(ctrl.Name or "").strip(),
                        control_type=ctype,
                        bbox=bbox,
                        window_title=window_title,
                        window_handle=window_handle,
                        control=ctrl,  # 缓存控件对象，供 get/set_element_text 使用
                    )
                )
        try:
            children = ctrl.GetChildren()
        except Exception:
            children = []
        for child in children:
            walk(child, depth + 1)

    walk(root_ctrl, 0)
    return elements


def _screen_size() -> tuple[int, int]:
    """屏幕物理像素尺寸（进程已设 DPI-aware，GetSystemMetrics 返回物理像素）。"""
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def list_ui_elements(max_elements: int = MAX_ELEMENTS) -> list[UIElement]:
    """枚举所有可见顶层窗口的可交互控件（前台窗口优先）。

    多窗口任务时 agent 需要能看到非前台窗口的元素（点击即激活该窗口），
    所以不只枚举前台。返回带稳定 id（1 起）的 ElementList（附带前台窗口标题）。
    预算分配：先逐窗口收集，再按"前台窗口优先 + 每窗口配额"截取，
    避免前台窗口的工具栏占满预算导致其他窗口/网页控件不可见。
    """
    import uiautomation as auto

    from screen_use.perception.screen import ensure_dpi_aware

    ensure_dpi_aware()

    screen_w, screen_h = _screen_size()
    screen_area = screen_w * screen_h

    try:
        fg = auto.GetForegroundControl()
        fg_handle = fg.NativeWindowHandle if fg else 0
    except Exception:
        fg_handle = 0

    # 收集可见顶层窗口，前台排最前
    root = auto.GetRootControl()
    windows = []
    for w in root.GetChildren():
        try:
            if w.ControlTypeName != "WindowControl" or w.IsOffscreen:
                continue
            rect = w.BoundingRectangle
            if rect.width() <= 0 or rect.height() <= 0:
                continue
            windows.append(w)
        except Exception:
            continue
    windows.sort(key=lambda w: 0 if _safe_handle(w) == fg_handle else 1)

    # 逐窗口收集（共享节点预算），并记录窗口标题
    visited = [0]
    per_window: list[list[UIElement]] = []
    foreground_title = ""
    for w in windows[:MAX_WINDOWS]:  # 前台优先，最多 5 个窗口
        try:
            title = (w.Name or "").strip() or "(无标题)"
        except Exception:
            title = "(无标题)"
        handle = _safe_handle(w)
        if handle == fg_handle:
            foreground_title = title
        per_window.append(_collect_from(w, title, handle, screen_area, visited))

    # 配额分配：前台窗口拿剩余大头，其余窗口均分配额
    elements: list[UIElement] = []
    n = len(per_window)
    if n:
        others_quota = max_elements // n
        fg_quota = max_elements - others_quota * (n - 1)
        for i, els in enumerate(per_window):
            quota = fg_quota if i == 0 else others_quota
            elements.extend(els[:quota])

    # 小控件优先（更可能是精确目标），再去重（同名同位置）
    elements.sort(key=lambda e: e.area)
    seen: set[tuple] = set()
    unique: list[UIElement] = []
    for el in elements:
        key = (el.name, el.control_type, el.bbox)
        if key not in seen:
            seen.add(key)
            unique.append(el)

    for i, el in enumerate(unique, start=1):
        el.id = i
    return ElementList(unique, foreground_title=foreground_title)
