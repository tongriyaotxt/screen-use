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
pyautogui.PAUSE = 0.02


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

    def _record(self, action: str, detail: str, ok: bool = True) -> ActionResult:
        result = ActionResult(ok=ok, action=action, detail=detail, dry_run=self.dry_run)
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
        """输入文本：
        - 纯 ASCII 走 pyautogui.typewrite（快，interval 控制键间间隔）
        - 含中文等非 ASCII 走剪贴板粘贴（typewrite 不支持，且避免偶发丢字符）
        粘贴后对 Edit 控件做轻量校验，不一致时返回 ok=False 并附 detail，
        上层应立即改用 set_element_text（UIA ValuePattern）而不是等下轮截图才发现。
        """
        if self.dry_run:
            return self._record("type_text", f"len={len(text)}")
        if text.isascii():
            pyautogui.typewrite(text, interval=interval)
            return self._record("type_text", f"len={len(text)} (typewrite)")
        error = self._paste_via_clipboard(text)
        if error:
            return self._record("type_text", error, ok=False)
        return self._record("type_text", f"len={len(text)} (clipboard)")

    def _paste_via_clipboard(self, text: str) -> str:
        """剪贴板粘贴：备份/恢复用户剪贴板，粘贴后轻量验证。返回错误描述，空串表示成功。"""
        import pyperclip

        try:
            backup = pyperclip.paste()  # 备份用户剪贴板
        except Exception:
            backup = None
        pyperclip.copy(text)
        time.sleep(0.05)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.05)
        if backup is not None:
            try:
                pyperclip.copy(backup)  # 恢复用户剪贴板
            except Exception:
                pass
        return self._verify_paste(text)

    @staticmethod
    def _verify_paste(text: str) -> str:
        """粘贴后轻量验证：焦点是 Edit 且支持 ValuePattern 时比对内容。

        返回空串表示通过或无法验证（无法验证时放行）；不一致返回差异描述。
        """
        try:
            import uiautomation as auto

            ctrl = auto.GetFocusedControl()
            if ctrl is None or ctrl.ControlTypeName != "EditControl":
                return ""
            vp = ctrl.GetValuePattern()
            if vp is None:
                return ""
            value = vp.Value or ""
            if text in value:
                return ""
            return (
                f"粘贴校验失败：期望包含 {text[:30]!r}，实际为 {value[:30]!r}"
                "（建议改用 set_element_text）"
            )
        except Exception:
            return ""  # 无法验证时放行

    def hotkey(self, *keys: str) -> ActionResult:
        if not self.dry_run:
            pyautogui.hotkey(*keys)
        return self._record("hotkey", "+".join(keys))

    def press(self, key: str) -> ActionResult:
        if not self.dry_run:
            pyautogui.press(key)
        return self._record("press", key)
