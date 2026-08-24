"""Executor 测试：dry_run 模式不触碰真实鼠标键盘。"""

from unittest.mock import patch

from screen_use.actions.executor import Executor


def test_dry_run_records_without_executing():
    with patch("screen_use.actions.executor.pyautogui") as mock_gui:
        ex = Executor(dry_run=True)
        ex.click(100, 200)
        ex.type_text("hello")
        ex.hotkey("ctrl", "s")
        ex.scroll(-3)
        mock_gui.click.assert_not_called()
        mock_gui.typewrite.assert_not_called()
        assert len(ex.history) == 4
        assert all(r.dry_run for r in ex.history)
        assert all(r.ok for r in ex.history)


def test_real_mode_calls_pyautogui():
    with patch("screen_use.actions.executor.pyautogui") as mock_gui:
        ex = Executor(dry_run=False)
        ex.click(10, 20)
        mock_gui.click.assert_called_once_with(10, 20, clicks=1, button="left")
        ex.double_click(10, 20)
        mock_gui.click.assert_called_with(10, 20, clicks=2, button="left")
        ex.right_click(10, 20)
        mock_gui.click.assert_called_with(10, 20, clicks=1, button="right")


def test_type_text_uses_clipboard():
    """所有文本（含纯 ASCII）统一走剪贴板粘贴，避免 typewrite 丢字符。"""
    with patch("screen_use.actions.executor.pyautogui") as mock_gui, \
         patch("pyperclip.copy") as mock_copy:
        Executor().type_text("Hello Agent")
        mock_gui.typewrite.assert_not_called()
        mock_copy.assert_called_once_with("Hello Agent")
        mock_gui.hotkey.assert_called_once_with("ctrl", "v")


def test_type_text_non_ascii_uses_clipboard():
    with patch("screen_use.actions.executor.pyautogui") as mock_gui, \
         patch("pyperclip.copy") as mock_copy:
        Executor().type_text("中文输入")
        mock_copy.assert_called_once_with("中文输入")
        mock_gui.hotkey.assert_called_once_with("ctrl", "v")
