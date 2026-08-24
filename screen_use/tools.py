"""SDK 门面：原子工具 + 高层工具的统一 Python API。

宿主 Agent 框架（LangChain / 自研）直接 import 本模块即可集成；
MCP Server（mcp_server.py）只是本模块的协议适配层。
"""

from __future__ import annotations

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

    # ================= 原子工具 =================

    def screenshot(self, annotate: bool = False) -> dict:
        """截屏。annotate=True 时叠加 SoM 编号框并附带元素列表。"""
        img, scale = screen.downscale(
            screen.screenshot(), self.settings.screenshot_max_size
        )
        elements: list[UIElement] = []
        if annotate:
            elements = self.list_ui_elements_raw()
            img = som.annotate(img, elements, scale)
        return {
            "image_base64": screen.to_jpeg_base64(img, self.settings.screenshot_jpeg_quality),
            "scale": scale,
            "width": img.size[0],
            "height": img.size[1],
            "elements": [el.to_dict() for el in elements],
        }

    def list_ui_elements_raw(self) -> list[UIElement]:
        self._elements = uia_tree.list_ui_elements()
        return self._elements

    def list_ui_elements(self) -> list[dict]:
        """枚举前台窗口的可交互控件（名称、类型、坐标）。零模型依赖。"""
        return [el.to_dict() for el in self.list_ui_elements_raw()]

    def click(self, x: int, y: int) -> dict:
        return self.executor.click(x, y).__dict__

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
                return self.executor.click(x, y).__dict__ | {"element": el.to_dict()}
        raise ValueError(
            f"元素 id={element_id} 不存在，请先调用 list_ui_elements 刷新元素列表"
        )

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

    def run_task(self, goal: str, max_steps: int = 20, on_step=None) -> dict:
        """视觉行为循环：observe → think → act → verify，直到任务完成。

        需要 VLM。on_step: 可选回调 fn(Step)，用于实时观察每一步。
        """
        from screen_use.agent import VisualLoop

        return VisualLoop(self, max_steps=max_steps).run(goal, on_step=on_step).to_dict()
