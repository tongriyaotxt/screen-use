"""UIA 控件树枚举：获取前台窗口的可交互元素。

坐标为物理像素（需先调用 screen.ensure_dpi_aware），与截图/点击坐标一致。
"""

from __future__ import annotations

import ctypes
import time
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
MAX_WINDOWS = 8   # 真实桌面顶层窗口常达 10+（2026-09-18 实测 13），5 会整窗丢失目标
MAX_DEPTH = 12
MAX_VISITED = 3000  # DFS 已访问节点预算：防止大页面（浏览器/IDE）秒级遍历
NODE_TIME_CHECK = 8  # 每走多少个节点检查一次时间预算（太密会自己变慢）

# 默认枚举时间预算（毫秒，0 = 不限）。前台窗口永远不限时（它是 agent 的工作面），
# 其余窗口按"学到的成本"从便宜到贵走，超预算即停——2026-09-19 实测：本机 437 个节点
# 里，一个音乐播放器 + Alienware 控制中心吃掉 1220/1416 ms（每节点 ~4ms 的 COM 往返），
# 而它们跟任务通常无关。预算只影响**后台**窗口的完整度，前台窗口永远完整。
DEFAULT_BUDGET_MS = 400

# 每个窗口的成本记忆（title → 毫秒 EMA）：用于把便宜窗口排在前面，
# 预算用尽时丢的是贵的（通常是无关的后台应用），不是便宜的目标窗口。
_WINDOW_COST: dict[str, float] = {}

# ---- 后台窗口元素表缓存（2026-09-19）----
# 前台窗口每步都要重新走（它是 agent 的工作面，且最便宜 ~50ms）；
# 后台窗口内容通常几分钟不变，却可能是慢 provider（实测两个后台窗口吃 86% 时间）。
# 命中条件：句柄 + 标题 + 窗口矩形都一致，且未过期。点之前另有实时 bbox 校验
# （element_still_valid）兜底，所以缓存不会导致点到别处。
WINDOW_CACHE_TTL_S = 2.0
_WINDOW_CACHE: dict[int, dict] = {}   # handle → {title, bbox, elements, ts}


def invalidate_window_cache(handles: list[int] | None = None) -> None:
    """清掉窗口缓存（None = 全清）。UI 变动、校验失败、或调用方要求新鲜时必须调。"""
    if handles is None:
        _WINDOW_CACHE.clear()
        return
    for h in handles:
        _WINDOW_CACHE.pop(h, None)


def cache_stats() -> dict:
    """缓存现状（诊断用）。"""
    return {"entries": len(_WINDOW_CACHE),
            "handles": sorted(_WINDOW_CACHE)}


def element_still_valid(el: UIElement, tol: int = 4) -> bool:
    """控件是否还在原地（读实时 bbox）。

    缓存复用 / 点击前的安全检查：UIA 元素引用在 UI 变动后可能失效（抛异常）
    或指向别处（矩形变了）。没有 control 对象（离线合成/测试）时不阻断。
    """
    ctrl = el.control
    if ctrl is None:
        return True
    try:
        rect = ctrl.BoundingRectangle
        if rect.width() <= 0 or rect.height() <= 0:
            return False
        current = (rect.left, rect.top, rect.right, rect.bottom)
    except Exception:
        return False
    return all(abs(a - b) <= tol for a, b in zip(current, el.bbox))


@dataclass
class UIElement:
    """一个可交互 UI 元素。bbox = (left, top, right, bottom)，物理像素。"""

    id: int
    name: str
    control_type: str
    bbox: tuple[int, int, int, int]
    window_title: str = ""
    window_handle: int = 0
    pid: int = 0
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
            "pid": self.pid,
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
    """list[UIElement] + 枚举时的前台窗口标题（当普通 list 用完全向后兼容）。

    timings: 枚举耗时拆分（哪个窗口吃了多少节点/多少毫秒）——观测是接入 Jev 后的
    新瓶颈（~0.7s/步 vs 决策 0.04s/步），优化前必须先看清钱花在哪。
    """

    def __init__(self, iterable=(), foreground_title: str = "", timings: dict | None = None):
        super().__init__(iterable)
        self.foreground_title = foreground_title
        self.timings = timings or {}


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
                  visited: list[int], max_visited: int = MAX_VISITED,
                  deadline: float | None = None) -> list[UIElement]:
    """从某个窗口的控件树收集可交互元素（深度优先）。

    visited: 跨窗口共享的已访问节点计数（[0] 为计数器），到 max_visited 即停，
    防止浏览器/IDE 等大页面的控件树导致秒级遍历。
    deadline: 绝对时间戳（perf_counter），到点就停止下降（已收集的元素保留）。
    """
    elements: list[UIElement] = []

    # 进程号**每个窗口读一次**（不是每个元素一次）：UIA 每次跨进程属性读取
    # 都可能是一趟 COM 往返，慢 provider 单节点就要 ~4ms（BENCH §6 实测）。
    # 用途：mock 合成器要判断"这个元素是不是我自己窗口的"
    # （2026-09-19 事故：观测把 owner 终端窗口的控件也算进了训练数据）。
    try:
        window_pid = int(root_ctrl.ProcessId)
    except Exception:
        window_pid = 0

    def walk(ctrl, depth: int) -> None:
        if visited[0] >= max_visited or depth > MAX_DEPTH:
            return
        if (deadline is not None and visited[0] % NODE_TIME_CHECK == 0
                and time.perf_counter() > deadline):
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
                        pid=window_pid,
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


def list_ui_elements(max_elements: int | None = MAX_ELEMENTS,
                     max_windows: int = MAX_WINDOWS,
                     max_visited: int = MAX_VISITED,
                     budget_ms: int | None = None,
                     force_full: bool = False,
                     cache_ttl_s: float | None = None) -> list[UIElement]:
    """枚举所有可见顶层窗口的可交互控件（前台窗口优先）。

    多窗口任务时 agent 需要能看到非前台窗口的元素（点击即激活该窗口），
    所以不只枚举前台。返回带稳定 id（1 起）的 ElementList（附带前台窗口标题）。
    预算分配：先逐窗口收集，再按"前台窗口优先 + 每窗口配额"截取，
    避免前台窗口的工具栏占满预算导致其他窗口/网页控件不可见。

    budget_ms: 枚举**总**时间预算（0/None 用 DEFAULT_BUDGET_MS）。前台窗口永不限时，
    其余窗口按学到的成本从便宜到贵走，超预算即停（已收集的元素保留，元素表可能变短）。
    max_elements: 元素总数上限；各窗口配额按它等比放大（合成环境里目标窗口不是前台时，
    默认 150 会让它只拿到 ~25 个名额，控件被切掉 → oracle 找不到目标）。
    force_full: 忽略缓存，全部窗口重走（UI 大变动后要求新鲜数据时用）。
    cache_ttl_s: 后台窗口元素表缓存有效期（秒，0=禁用；None 用 WINDOW_CACHE_TTL_S）。
    实测拆分见 `el.timings`。并行遍历试过（2026-09-19）：本机无收益（UIA 跨进程调用
    在 broker 侧仍然串行），已回退。
    """
    import time

    import uiautomation as auto

    from screen_use.perception.screen import ensure_dpi_aware

    if max_elements is None:
        max_elements = MAX_ELEMENTS
    if budget_ms is None:
        budget_ms = DEFAULT_BUDGET_MS
    if cache_ttl_s is None:
        cache_ttl_s = WINDOW_CACHE_TTL_S
    deadline = (time.perf_counter() + budget_ms / 1000) if budget_ms else None

    t_start = time.perf_counter()
    ensure_dpi_aware()

    screen_w, screen_h = _screen_size()
    screen_area = screen_w * screen_h

    try:
        fg = auto.GetForegroundControl()
        fg_handle = fg.NativeWindowHandle if fg else 0
    except Exception:
        fg_handle = 0

    # 收集可见顶层窗口，前台排最前（顺便记下矩形：既做缓存键，也避免重复 COM 读取）
    root = auto.GetRootControl()
    windows = []
    rects: dict[int, tuple[int, int, int, int]] = {}
    for w in root.GetChildren():
        try:
            if w.ControlTypeName != "WindowControl" or w.IsOffscreen:
                continue
            rect = w.BoundingRectangle
            if rect.width() <= 0 or rect.height() <= 0:
                continue
            windows.append(w)
            rects[_safe_handle(w)] = (rect.left, rect.top, rect.right, rect.bottom)
        except Exception:
            continue
    windows.sort(key=lambda w: 0 if _safe_handle(w) == fg_handle else 1)
    windows_ms = (time.perf_counter() - t_start) * 1000

    # 后台窗口按学到的成本升序：预算用尽时丢的是贵的（通常与任务无关）
    foreground = [w for w in windows[:max_windows] if _safe_handle(w) == fg_handle][:1]
    background = [w for w in windows[:max_windows] if _safe_handle(w) != fg_handle]
    background.sort(key=lambda w: _window_cost(w))

    # 逐窗口收集（共享节点预算 + 总时间预算；前台窗口不吃截止时间、也不吃缓存）
    visited = [0]
    per_window: list[list[UIElement]] = []
    foreground_title = ""
    per_window_stats: list[dict] = []
    skipped: list[str] = []
    reused: list[str] = []
    for w in foreground + background:
        try:
            title = (w.Name or "").strip() or "(无标题)"
        except Exception:
            title = "(无标题)"
        handle = _safe_handle(w)
        is_fg = handle == fg_handle
        if is_fg:
            foreground_title = title

        # 后台窗口缓存命中（句柄+标题+矩形一致且未过期）→ 直接复用，跳过整棵遍历
        if not is_fg and not force_full and cache_ttl_s:
            entry = _WINDOW_CACHE.get(handle)
            if (entry is not None and entry["title"] == title
                    and entry["bbox"] == rects.get(handle)
                    and time.perf_counter() - entry["ts"] <= cache_ttl_s):
                per_window.append(entry["elements"])
                reused.append(title[:40])
                per_window_stats.append({
                    "title": title[:40], "nodes": 0,
                    "elements": len(entry["elements"]), "ms": 0.0,
                    "foreground": False, "cut": False, "cached": True,
                })
                continue

        if not is_fg and deadline is not None and time.perf_counter() > deadline:
            skipped.append(title[:40])  # 预算用尽：跳过（timings 里如实记账）
            continue
        t_w = time.perf_counter()
        nodes_before = visited[0]
        els = _collect_from(w, title, handle, screen_area, visited, max_visited,
                            deadline=None if is_fg else deadline)
        spent = (time.perf_counter() - t_w) * 1000
        _remember_cost(title, spent)
        _WINDOW_CACHE[handle] = {"title": title, "bbox": rects.get(handle),
                                 "elements": els, "ts": time.perf_counter()}
        per_window_stats.append({
            "title": title[:40], "nodes": visited[0] - nodes_before,
            "elements": len(els), "ms": round(spent, 1),
            "foreground": is_fg,
            "cut": bool(not is_fg and deadline is not None and time.perf_counter() > deadline),
        })
        per_window.append(els)

    # 清掉已经不在桌面上的窗口缓存，防句柄复用导致的张冠李戴
    invalidate_window_cache([h for h in list(_WINDOW_CACHE) if h not in rects])

    # 配额分配：前台窗口保底一半，其余窗口均分另一半。
    # 每窗口内先去重再按面积升序截取：重复项不吃配额，小控件（按钮）优先于大容器——
    # 避免计算器等窗口的数字按钮被同名重复项/大面板挤掉（2026-09-18 实测 7/8/9 丢失事故）。
    elements: list[UIElement] = []
    n = len(per_window)
    if n:
        fg_quota = max_elements // 2
        others_quota = (max_elements - fg_quota) // max(n - 1, 1)
        for i, els in enumerate(per_window):
            quota = fg_quota if i == 0 else others_quota
            els = _dedup_by_identity(els)
            els.sort(key=lambda e: e.area)
            elements.extend(els[:quota])

    # 小控件优先（更可能是精确目标），再去重（同名同位置）
    elements.sort(key=lambda e: e.area)
    unique = _dedup_by_identity(elements)

    for i, el in enumerate(unique, start=1):
        el.id = i
    timings = {
        "total_ms": round((time.perf_counter() - t_start) * 1000, 1),
        "windows_ms": round(windows_ms, 1),
        "nodes": visited[0],
        "n_windows": len(per_window),
        "budget_ms": budget_ms,
        "skipped": skipped,
        "reused": reused,
        "cache_entries": len(_WINDOW_CACHE),
        "windows": per_window_stats,
    }
    return ElementList(unique, foreground_title=foreground_title, timings=timings)


def _window_cost(win) -> float:
    """窗口的历史枚举成本（毫秒，EMA）；没见过的按 0 排前面（先便宜后贵）。"""
    try:
        title = (win.Name or "").strip() or "(无标题)"
    except Exception:
        return 0.0
    return _WINDOW_COST.get(title, 0.0)


def _remember_cost(title: str, spent_ms: float) -> None:
    old = _WINDOW_COST.get(title)
    _WINDOW_COST[title] = spent_ms if old is None else old * 0.7 + spent_ms * 0.3


def _dedup_by_identity(elements: list[UIElement]) -> list[UIElement]:
    """按 (名称, 类型, 位置) 去重，保持原顺序。"""
    seen: set[tuple] = set()
    unique: list[UIElement] = []
    for el in elements:
        key = (el.name, el.control_type, el.bbox)
        if key not in seen:
            seen.add(key)
            unique.append(el)
    return unique
