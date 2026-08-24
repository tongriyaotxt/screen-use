"""动作执行层：pyautogui 封装。

- FAILSAFE：鼠标甩到屏幕左上角立即抛 FailSafeException 急停
- dry_run：只记录不执行（测试/调试）
- 所有坐标为物理像素（与 screen/uia_tree 一致）
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


@dataclass
class ActionResult:
    ok: bool
    action: str
    detail: str = ""
    dry_run: bool = False


class Executor:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.history: list[ActionResult] = []

    def _record(self, action: str, detail: str) -> ActionResult:
        result = ActionResult(ok=True, action=action, detail=detail, dry_run=self.dry_run)
        self.history.append(result)
        return result

    # ---- 鼠标 ----

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> ActionResult:
        detail = f"({x}, {y}) button={button} clicks={clicks}"
        if not self.dry_run:
            pyautogui.click(x, y, clicks=clicks, button=button)
        return self._record("click", detail)

    def double_click(self, x: int, y: int) -> ActionResult:
        return self.click(x, y, clicks=2)

    def right_click(self, x: int, y: int) -> ActionResult:
        return self.click(x, y, button="right")

    def move_to(self, x: int, y: int) -> ActionResult:
        if not self.dry_run:
            pyautogui.moveTo(x, y)
        return self._record("move_to", f"({x}, {y})")

    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> ActionResult:
        """clicks 正数向上滚，负数向下滚。"""
        if not self.dry_run:
            if x is not None and y is not None:
                pyautogui.moveTo(x, y)
            pyautogui.scroll(clicks)
        return self._record("scroll", f"clicks={clicks}")

    # ---- 键盘 ----

    def type_text(self, text: str, interval: float = 0.02) -> ActionResult:
        """输入文本。统一走剪贴板粘贴：
        - 支持中文等非 ASCII 字符
        - 避免 pyautogui.typewrite 偶发丢字符（如空格）的问题
        """
        if not self.dry_run:
            self._paste_via_clipboard(text)
        return self._record("type_text", f"len={len(text)}")

    def _paste_via_clipboard(self, text: str) -> None:
        import pyperclip

        pyperclip.copy(text)
        time.sleep(0.1)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.1)

    def hotkey(self, *keys: str) -> ActionResult:
        if not self.dry_run:
            pyautogui.hotkey(*keys)
        return self._record("hotkey", "+".join(keys))

    def press(self, key: str) -> ActionResult:
        if not self.dry_run:
            pyautogui.press(key)
        return self._record("press", key)
