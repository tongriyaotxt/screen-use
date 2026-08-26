"""SDK 门面：原子工具 + 高层工具的统一 Python API。

宿主 Agent 框架（LangChain / 自研）直接 import 本模块即可集成；
MCP Server（mcp_server.py）只是本模块的协议适配层。
"""

from __future__ import annotations

import time
from typing import Callable

from screen_use.actions.executor import Executor
from screen_use.config import Settings, get_settings
from screen_use.memory import ExperienceStore
from screen_use.perception import screen, som, uia_tree
from screen_use.perception.uia_tree import UIElement
from screen_use.providers.base import VisionProvider
from screen_use.providers.openai_compat import OpenAICompatProvider


class ScreenUse:
    """桌面自动化工具集。

    Args:
        dry_run: 只记录动作不执行（测试/调试）
        settings: 配置，默认从 .env 读取
        provider: 自定义 VLM Provider，默认按配置创建（VLM 未配置时高层视觉工具不可用）
        confirm_callback: 危险动作确认钩子，签名为 (动作描述) -> bool，返回 False 则取消执行
    """

    def __init__(
        self,
        dry_run: bool = False,
        settings: Settings | None = None,
        provider: VisionProvider | None = None,
        confirm_callback: Callable[[str], bool] | None = None,
        memory: ExperienceStore | None = None,
    ):
        self.settings = settings or get_settings()
        self.executor = Executor(dry_run=dry_run)
        self.confirm_callback = confirm_callback
        self._provider = provider
        self.memory = memory or ExperienceStore()  # 元学习：经验与词汇记忆
        self._elements: list[UIElement] = []  # 最近一次 list_ui_elements 的缓存
        self._foreground_title = ""           # 最近一次枚举时的前台窗口标题
        self._last_scale = 1.0                # 最近一次 screenshot 的坐标换算比例

    # ---- Provider 懒加载 ----

    @property
    def provider(self) -> VisionProvider:
        if self._provider is None:
            if not self.settings.vlm_available:
                raise RuntimeError(
                    "VLM 未配置：请在 .env 中设置 VISION_API_KEY / VISION_MODEL，"
                    "或使用本地 Ollama（VISION_PROVIDER=ollama）。"
                    "原子工具和 UIA 匹配不依赖 VLM，仍可正常使用。"
                )
            self._provider = OpenAICompatProvider(self.settings)
        return self._provider

    def _confirm(self, description: str) -> None:
        if self.confirm_callback and not self.confirm_callback(description):
            raise PermissionError(f"操作被确认回调取消: {description}")

    @staticmethod
    def _activate_window(el: UIElement) -> None:
        """点击前把元素所属窗口激活到前台（防止窗口被遮挡时点穿到别的窗口）。"""
        if not el.window_handle:
            return
        import ctypes
        import time

        user32 = ctypes.windll.user32
        hwnd = el.window_handle
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE（若在最小化）
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.15)

    # ================= 原子工具 =================

    def screenshot(self, annotate: bool = False,
                   max_size: int | None = None, quality: int | None = None) -> dict:
        """截屏。annotate=True 时叠加 SoM 编号框并附带元素列表。

        max_size / quality 缺省读配置；JPEG 字节数超 screenshot_max_bytes 时
        自动降 quality 重编码，保证 MCP 输出不被宿主截断。
        """
        if max_size is None:
            max_size = self.settings.screenshot_max_size
        if quality is None:
            quality = self.settings.screenshot_jpeg_quality
        img, scale = screen.downscale(screen.screenshot(), max_size)
        self._last_scale = scale
        elements: list[UIElement] = []
        if annotate:
            elements = self.list_ui_elements_raw()
            img = som.annotate(img, elements, scale)
        b64, used_quality = screen.to_jpeg_base64_capped(
            img, quality, self.settings.screenshot_max_bytes
        )
        return {
            "image_base64": b64,
            "scale": scale,
            "width": img.size[0],
            "height": img.size[1],
            "quality": used_quality,
            "foreground_title": self._foreground_title,
            "elements": [el.to_dict() for el in elements],
        }

    def list_ui_elements_raw(self) -> list[UIElement]:
        result = uia_tree.list_ui_elements()
        self._foreground_title = getattr(result, "foreground_title", "")
        self._elements = list(result)
        return self._elements

    def list_ui_elements(self) -> dict:
        """枚举前台窗口的可交互控件（名称、类型、坐标）。零模型依赖。

        返回 {"foreground_title": 前台窗口标题, "elements": [...]}。
        """
        elements = [el.to_dict() for el in self.list_ui_elements_raw()]
        return {"foreground_title": self._foreground_title, "elements": elements}

    def click(self, x: int, y: int) -> dict:
        return self.executor.click(x, y).__dict__

    def click_scaled(self, x: int, y: int) -> dict:
        """点击截图上的缩放坐标（内部乘最近一次 screenshot 的 scale 换算回物理像素）。"""
        return self.click(int(x * self._last_scale), int(y * self._last_scale))

    def double_click(self, x: int, y: int) -> dict:
        return self.executor.double_click(x, y).__dict__

    def right_click(self, x: int, y: int) -> dict:
        return self.executor.right_click(x, y).__dict__

    def click_element_id(self, element_id: int) -> dict:
        """点击 list_ui_elements 返回的元素 id（使用最近一次枚举的缓存）。"""
        for el in self._elements:
            if el.id == element_id:
                x, y = el.center
                self._confirm(f"点击元素 #{element_id} [{el.control_type}] {el.name!r} @({x},{y})")
                self._activate_window(el)
                return self.executor.click(x, y).__dict__ | {"element": el.to_dict()}
        raise ValueError(
            f"元素 id={element_id} 不存在，请先调用 list_ui_elements 刷新元素列表"
        )

    def _find_cached(self, element_id: int) -> UIElement:
        for el in self._elements:
            if el.id == element_id:
                return el
        raise ValueError(
            f"元素 id={element_id} 不存在，请先调用 list_ui_elements 刷新元素列表"
        )

    def get_element_text(self, element_id: int) -> dict:
        """读控件文本（优先 UIA ValuePattern，退回控件名称）。"""
        el = self._find_cached(element_id)
        try:
            text = uia_tree.get_element_text(el)
        except Exception as e:
            return {"ok": False, "text": "", "detail": str(e)}
        return {"ok": True, "text": text}

    def set_element_text(self, element_id: int, text: str) -> dict:
        """写控件文本：走 UIA ValuePattern.SetValue，绕过禁粘贴的输入框。"""
        el = self._find_cached(element_id)
        self._confirm(f"设置元素 #{element_id} [{el.control_type}] {el.name!r} 的文本: {text[:50]!r}")
        self._activate_window(el)
        try:
            uia_tree.set_element_text(el, text)
        except ValueError as e:
            return {
                "ok": False,
                "detail": f"{e}（可回退：click_element_id 聚焦后 type_text）",
            }
        return {"ok": True, "detail": f"已设置元素 #{element_id} 文本 len={len(text)}"}

    def type_text(self, text: str) -> dict:
        self._confirm(f"输入文本: {text[:50]!r}")
        return self.executor.type_text(text).__dict__

    def hotkey(self, *keys: str) -> dict:
        self._confirm(f"快捷键: {'+'.join(keys)}")
        return self.executor.hotkey(*keys).__dict__

    def press(self, key: str) -> dict:
        return self.executor.press(key).__dict__

    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> dict:
        return self.executor.scroll(clicks, x, y).__dict__

    # ================= 批量动作 =================

    def do_actions(self, actions: list[dict], interval: float = 0.3,
                   shot_max_size: int | None = None, shot_quality: int | None = None) -> dict:
        """批量执行一串动作，每步间 sleep interval 秒，最后返回一次截图（体积受控）。

        支持的动作：click / double_click / right_click / type_text / press /
        hotkey / scroll / click_element_id / set_element_text。
        让宿主一次调用完成 "点输入框→输入→Tab→输入→回车" 这类序列。
        某步出错即中止（避免在错误状态下继续盲操作）。
        """
        results = []
        for i, spec in enumerate(actions):
            action = spec.get("action", "")
            try:
                r = self._dispatch_action(action, spec)
                ok = bool(r.get("ok", True))
                detail = r.get("detail", "")
            except Exception as e:
                ok, detail = False, str(e)
            results.append({"step": i, "action": action, "ok": ok, "detail": detail})
            if not ok:
                break
            if i < len(actions) - 1:
                time.sleep(interval)
        shot = self.screenshot(max_size=shot_max_size, quality=shot_quality)
        return {"results": results, "screenshot": shot}

    def _dispatch_action(self, action: str, spec: dict) -> dict:
        """do_actions 的单步分发。"""
        if action == "click":
            return self.click(int(spec["x"]), int(spec["y"]))
        if action == "double_click":
            return self.double_click(int(spec["x"]), int(spec["y"]))
        if action == "right_click":
            return self.right_click(int(spec["x"]), int(spec["y"]))
        if action == "type_text":
            return self.type_text(str(spec["text"]))
        if action == "press":
            return self.press(str(spec["key"]))
        if action == "hotkey":
            keys = spec["keys"]
            if isinstance(keys, str):
                keys = keys.split("+")
            return self.hotkey(*keys)
        if action == "scroll":
            return self.scroll(int(spec["clicks"]), spec.get("x"), spec.get("y"))
        if action == "click_element_id":
            return self.click_element_id(int(spec["element_id"]))
        if action == "set_element_text":
            return self.set_element_text(int(spec["element_id"]), str(spec["text"]))
        raise ValueError(f"不支持的动作: {action!r}")

    # ================= 高层工具（策略链） =================

    def find_element(self, description: str) -> dict:
        """按自然语言描述定位元素。

        策略链：
          0. 元学习词汇映射（以往任务学到的 描述→控件名）
          1. UIA 文本匹配（零模型依赖）
          2. SoM + VLM
        返回 {found, method, element}。
        """
        elements = self.list_ui_elements_raw()

        # 第 0 级：词汇记忆（之前成功定位过的描述直接命中）
        learned = self.memory.recall_mapping(description)
        if learned:
            for el in elements:
                if el.name == learned["name"]:
                    return {"found": True, "method": "learned", "element": el.to_dict()}

        # 第 1 级：UIA 文本匹配（零模型依赖）
        hit = self._uia_match(elements, description)
        if hit is not None:
            el, method = hit
            self.memory.learn_mapping(description, el.name, el.control_type)
            return {"found": True, "method": method, "element": el.to_dict()}

        # 第 2 级：SoM + VLM
        if not self.settings.vlm_available:
            return {
                "found": False,
                "method": "none",
                "error": (
                    f"UIA 未找到 {description!r}，且 VLM 未配置，无法继续视觉定位。"
                    "可配置 VLM（.env）或改用 screenshot/list_ui_elements 由宿主自行定位。"
                ),
            }
        img, scale = screen.downscale(
            screen.screenshot(), self.settings.screenshot_max_size
        )
        annotated = som.annotate(img, elements, scale)
        b64 = screen.to_jpeg_base64(annotated, self.settings.screenshot_jpeg_quality)
        try:
            picked_id = self.provider.pick_element(b64, som.legend(elements), description)
        except RuntimeError as e:
            return {"found": False, "method": "vlm", "error": str(e)}
        for el in elements:
            if el.id == picked_id:
                self.memory.learn_mapping(description, el.name, el.control_type)
                return {"found": True, "method": "vlm", "element": el.to_dict()}
        return {"found": False, "method": "vlm", "error": f"VLM 认为不存在 {description!r}"}

    @staticmethod
    def _uia_match(elements: list[UIElement], description: str) -> tuple[UIElement, str] | None:
        """UIA 文本匹配：先精确后模糊。"""
        desc = description.strip().lower()
        if not desc:
            return None
        named = [el for el in elements if el.name]
        # 精确匹配
        for el in named:
            if el.name.strip().lower() == desc:
                return el, "uia_exact"
        # 模糊匹配（双向包含）
        fuzzy = [
            el for el in named
            if desc in el.name.strip().lower() or el.name.strip().lower() in desc
        ]
        if fuzzy:
            fuzzy.sort(key=lambda e: e.area)  # 取最小（最精确）的
            return fuzzy[0], "uia_fuzzy"
        return None

    def click_element(self, description: str) -> dict:
        """按自然语言描述定位并点击，一步完成。"""
        result = self.find_element(description)
        if not result.get("found"):
            return result
        x, y = result["element"]["center"]
        self._confirm(
            f"点击 {description!r}（{result['method']} 命中: "
            f"[{result['element']['control_type']}] {result['element']['name']!r} @({x},{y})）"
        )
        # 激活元素所属窗口（防止被遮挡时点穿）
        for el in self._elements:
            if el.id == result["element"]["id"]:
                self._activate_window(el)
                break
        action = self.executor.click(x, y)
        return result | {"action": action.__dict__}

    def read_screen(self, question: str) -> str:
        """就当前屏幕内容提问（硬性需要 VLM）。"""
        provider = self.provider  # VLM 未配置时在此直接报错，不做无谓截图
        _, b64, _ = screen.capture_for_model(
            self.settings.screenshot_max_size, self.settings.screenshot_jpeg_quality
        )
        return provider.ask_about_screen(b64, question)

    # ================= 视觉行为循环 =================

    def run_task(self, goal: str, max_steps: int = 40, on_step=None) -> dict:
        """视觉行为循环：observe → think → act → verify，直到任务完成。

        需要 VLM。on_step: 可选回调 fn(Step)，用于实时观察每一步。
        """
        from screen_use.agent import VisualLoop

        return VisualLoop(self, max_steps=max_steps).run(goal, on_step=on_step).to_dict()
