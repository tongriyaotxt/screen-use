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


def test_type_text_ascii_uses_typewrite():
    """纯 ASCII 走 typewrite（快），不走剪贴板。"""
    with patch("screen_use.actions.executor.pyautogui") as mock_gui, \
         patch("pyperclip.copy") as mock_copy:
        Executor().type_text("Hello Agent")
        mock_gui.typewrite.assert_called_once_with("Hello Agent", interval=0.02)
        mock_copy.assert_not_called()


def test_type_text_non_ascii_uses_clipboard():
    """非 ASCII 走剪贴板粘贴，且备份/恢复用户剪贴板（copy 两次：内容 + 恢复）。"""
    with patch("screen_use.actions.executor.pyautogui") as mock_gui, \
         patch("pyperclip.copy") as mock_copy, \
         patch("pyperclip.paste", return_value="用户原剪贴板"), \
         patch.object(Executor, "_verify_paste", return_value=""):
        Executor().type_text("中文输入")
        assert mock_copy.call_args_list[0].args == ("中文输入",)
        assert mock_copy.call_args_list[1].args == ("用户原剪贴板",)  # 粘贴后恢复
        mock_gui.hotkey.assert_called_once_with("ctrl", "v")


def test_type_text_paste_verify_failure():
    """粘贴校验失败时返回 ok=False 并附 detail（上层可立即换 set_element_text）。"""
    with patch("screen_use.actions.executor.pyautogui"), \
         patch("pyperclip.copy"), \
         patch("pyperclip.paste", return_value=""), \
         patch.object(Executor, "_verify_paste", return_value="粘贴校验失败：期望包含 'abc'"):
        result = Executor().type_text("中文")
    assert result.ok is False
    assert "粘贴校验失败" in result.detail


def test_verify_paste_passes_when_unverifiable():
    """无法获取焦点控件/非 Edit 时放行（返回空串）。"""
    with patch("uiautomation.GetFocusedControl", return_value=None):
        assert Executor._verify_paste("任意文本") == ""


def _fake_edit(value: str):
    from unittest.mock import MagicMock

    ctrl = MagicMock()
    ctrl.ControlTypeName = "EditControl"
    ctrl.GetValuePattern.return_value.Value = value
    return ctrl


def test_verify_paste_match_and_mismatch():
    """焦点是 Edit 且支持 ValuePattern 时比对内容：包含则放行，不一致返回差异描述。"""
    with patch("uiautomation.GetFocusedControl", return_value=_fake_edit("已输入中文内容")):
        assert Executor._verify_paste("中文内容") == ""
    with patch("uiautomation.GetFocusedControl", return_value=_fake_edit("")):
        detail = Executor._verify_paste("中文内容")
        assert "粘贴校验失败" in detail and "set_element_text" in detail
