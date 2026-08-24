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

MAX_ELEMENTS = 50
MAX_DEPTH = 12


@dataclass
class UIElement:
    """一个可交互 UI 元素。bbox = (left, top, right, bottom)，物理像素。"""

    id: int
    name: str
    control_type: str
    bbox: tuple[int, int, int, int]
    window_title: str = ""
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


def list_ui_elements(max_elements: int = MAX_ELEMENTS) -> list[UIElement]:
    """枚举前台窗口的可交互控件，按面积从小到大排序（小控件通常是精确目标）。

    返回带稳定 id（1 起）的 UIElement 列表，id 在当前列表内唯一。
    """
    import uiautomation as auto

    from screen_use.perception.screen import ensure_dpi_aware, screenshot

    ensure_dpi_aware()

    root = auto.GetForegroundControl()
    if root is None:
        # 前台无窗口时退化为桌面
        root = auto.GetRootControl()
    window_title = getattr(root, "Name", "") or ""

    screen_w, screen_h = screenshot().size
    screen_area = screen_w * screen_h

    elements: list[UIElement] = []

    def walk(ctrl, depth: int) -> None:
        if len(elements) >= max_elements or depth > MAX_DEPTH:
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
                        id=0,  # 排序后统一编号
                        name=(ctrl.Name or "").strip(),
                        control_type=ctype,
                        bbox=bbox,
                        window_title=window_title,
                    )
                )
        try:
            children = ctrl.GetChildren()
        except Exception:
            children = []
        for child in children:
            walk(child, depth + 1)

    walk(root, 0)

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
