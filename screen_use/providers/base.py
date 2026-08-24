"""VisionProvider 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod


class VisionProvider(ABC):
    """多模态视觉模型接口。所有实现必须支持 OpenAI 兼容的 chat.completions 协议。"""

    @abstractmethod
    def pick_element(self, image_base64: str, legend: str, goal: str) -> int:
        """从 SoM 标注图中选出目标元素编号。

        Args:
            image_base64: 标注后的截图（JPEG base64）
            legend: 编号 → 控件名/类型 的图例文本
            goal: 用户要找的目标描述，如 "保存按钮"

        Returns:
            元素编号（1 起）。找不到时返回 -1。
        """

    @abstractmethod
    def ask_about_screen(self, image_base64: str, question: str) -> str:
        """就屏幕内容提问（读屏）。"""
