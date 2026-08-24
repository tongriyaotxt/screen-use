"""UIA 控件树枚举：获取前台窗口的可交互元素。

坐标为物理像素（需先调用 screen.ensure_dpi_aware），与截图/点击坐标一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field

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

MAX_ELEMENTS = 60
MAX_WINDOWS = 5
MAX_DEPTH = 12


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
                  elements: list, budget: int) -> None:
    """从某个窗口的控件树收集可交互元素（深度优先，受 budget 限制）。"""

    def walk(ctrl, depth: int) -> None:
        if len(elements) >= budget or depth > MAX_DEPTH:
            return
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
                    )
                )
        try:
            children = ctrl.GetChildren()
        except Exception:
            children = []
        for child in children:
            walk(child, depth + 1)

    walk(root_ctrl, 0)


def list_ui_elements(max_elements: int = MAX_ELEMENTS) -> list[UIElement]:
    """枚举所有可见顶层窗口的可交互控件（前台窗口优先）。

    多窗口任务时 agent 需要能看到非前台窗口的元素（点击即激活该窗口），
    所以不只枚举前台。返回带稳定 id（1 起）的列表。
    """
    import uiautomation as auto

    from screen_use.perception.screen import ensure_dpi_aware, screenshot

    ensure_dpi_aware()

    screen_w, screen_h = screenshot().size
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

    elements: list[UIElement] = []
    for w in windows[:MAX_WINDOWS]:  # 前台优先，最多 5 个窗口
        try:
            title = (w.Name or "").strip() or "(无标题)"
        except Exception:
            title = "(无标题)"
        _collect_from(w, title, _safe_handle(w), screen_area, elements, max_elements)
        if len(elements) >= max_elements:
            break

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
    return unique
