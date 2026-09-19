"""mock 窗口生命周期护栏（offscreen，不开窗）。

2026-09-19 事故：v3 量产到第 103 回合时，owner 报「哪个 mock 卡住了，显示未响应」。
现场是 `MockUI-mock-v3-7-0049` —— **第 49 回合的确认框**。三个因素叠加：

1. 确认框是 `QDialog(self)`，而且**故意与主窗口同标题**（`setWindowTitle(self.windowTitle())`，
   为了让观测按标题 tag 捕获弹窗里的控件）→ 从外面看就像主窗口卡住了；
2. 有些范式的 oracle 流程最后一步把确认框留在**打开**状态（如「点删除→确认框弹出」就收尾）；
3. `win.close()` 只隐藏主窗口，**对话框是独立顶层窗口**，照样赖在屏幕上；
   而生成器进 idle-gate 后 sleep 会阻塞 Qt 事件循环 → Windows 给它标上「(未响应)」。

修法两层：`MockWindow.teardown()` 按层收（弹层→确认框→本体），
`mock_env.sweep_stale_windows()` 在回合边界兜底扫。
"""

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _dialog_spec(rng) -> dict:
    """跟 mock_env.t_dialog_confirm 同构：点「删除」会弹出确认框。"""
    import mock_env

    spec = mock_env._base_spec(rng)
    spec["dialog_on_click"] = ("删除", ["确定", "取消"])
    spec["buttons"] = [("删除", True)] + [(b, e) for b, e in spec["buttons"] if b != "删除"]
    return spec


def _visible_mock_windows(app):
    return [w for w in app.topLevelWidgets()
            if w.isVisible() and w.windowTitle().startswith("MockUI")]


def test_confirm_dialog_is_its_own_top_level_window_with_same_title(monkeypatch):
    """先把事故前提钉死：确认框是独立顶层窗口，且标题与主窗口一模一样。"""
    app = _app(monkeypatch)
    import mock_env
    from PySide6.QtWidgets import QDialog

    rng = random.Random(5)
    win = mock_env.MockWindow(rng, "mock-v3-7-0049", _dialog_spec(rng))
    win.show()
    app.processEvents()
    win.buttons["删除"].click()
    app.processEvents()

    dialogs = [w for w in _visible_mock_windows(app) if isinstance(w, QDialog)]
    assert dialogs, "确认框没弹出来，后面的测试等于没跑"
    assert dialogs[0].windowTitle() == "MockUI-mock-v3-7-0049"  # 与主窗口同名
    win.teardown()
    app.processEvents()


def test_teardown_takes_the_dialog_with_it(monkeypatch):
    """`win.close()` 时代会漏掉确认框；teardown 必须一个窗口都不留。"""
    app = _app(monkeypatch)
    import mock_env

    rng = random.Random(6)
    win = mock_env.MockWindow(rng, "mock-v3-7-0049", _dialog_spec(rng))
    win.show()
    app.processEvents()
    win.buttons["删除"].click()
    app.processEvents()
    assert _visible_mock_windows(app), "前提不成立：确认框没开"

    win.teardown()
    app.processEvents()

    left = _visible_mock_windows(app)
    assert not left, f"收摊后仍留下窗口：{[w.windowTitle() for w in left]}"


def test_sweep_spares_the_current_round_and_clears_strays(monkeypatch):
    """兜底清扫：只清残留，不动本回合正在用的窗口（回合中调会误杀弹窗）。"""
    app = _app(monkeypatch)
    import mock_env

    rng = random.Random(7)
    current = mock_env.MockWindow(rng, "current", mock_env._base_spec(rng))
    current.show()
    stray = mock_env.MockWindow(rng, "stray", mock_env._base_spec(rng))
    stray.show()
    app.processEvents()
    assert len(_visible_mock_windows(app)) == 2

    assert mock_env.sweep_stale_windows(keep=current) == 1
    app.processEvents()

    left = [w.windowTitle() for w in _visible_mock_windows(app)]
    assert left == ["MockUI-current"], f"扫错了对象：{left}"
    current.teardown()
    app.processEvents()


def test_sweep_catches_a_dialog_whose_parent_is_gone(monkeypatch):
    """最坏情况：确认框的主人已经没了，兜底清扫也要能靠标题前缀把它收掉。"""
    app = _app(monkeypatch)
    import mock_env
    from PySide6.QtWidgets import QDialog

    rng = random.Random(8)
    owner = mock_env.MockWindow(rng, "gone", _dialog_spec(rng))
    owner.show()
    app.processEvents()
    owner.buttons["删除"].click()
    app.processEvents()
    dlg = next(w for w in _visible_mock_windows(app) if isinstance(w, QDialog))
    dlg.setParent(None)  # 模拟主人已销毁、确认框被孤儿化
    dlg.show()           # setParent 会自动隐藏，得再显形才能复现"赖在屏幕上"
    owner.close()
    app.processEvents()
    assert _visible_mock_windows(app), "前提不成立：孤儿确认框不在屏幕上"

    assert mock_env.sweep_stale_windows() >= 1
    app.processEvents()
    assert not _visible_mock_windows(app)


# ---- 无名图标控件：Name 必须仍为空，同时像素里得有图形 ----
# owner 2026-09-19 报「我没看到纯图标」——查实 v3 的"图标按钮"就是 QPushButton("")，
# 空白方块，一个图形都没画（真实桌面那批是窗口按钮 ─ □ ✕，实测 bbox 69×60）。

def test_blank_named_buttons_carry_a_real_glyph(monkeypatch):
    app = _app(monkeypatch)
    import mock_env
    from PySide6.QtWidgets import QPushButton

    rng = random.Random(11)
    spec = mock_env._base_spec(rng)
    spec["buttons"] = [("保存", True), ("", True), ("", True), ("取消", True)]
    win = mock_env.MockWindow(rng, "icons", spec)
    try:
        blanks = [b for b in win.findChildren(QPushButton) if not b.text().strip()]
        named = [b for b in win.findChildren(QPushButton) if b.text().strip()]
        assert len(blanks) == 2, "前提不成立：没造出无名按钮"
        for b in blanks:
            icon = b.icon()
            assert not icon.isNull(), "无名按钮没有图标 = 又变回空白方块"
            assert not icon.pixmap(16, 16).toImage().isNull(), "图标是空的"
        for b in named:
            assert b.icon().isNull(), "有名字的按钮不该被塞图标（会改变原有外观）"
    finally:
        win.teardown()
        app.processEvents()


def test_glyph_shapes_are_distinct(monkeypatch):
    """不同 shape 得画出不同图案，否则"图标"只是同一个方块复制粘贴。"""
    app = _app(monkeypatch)
    import mock_env

    seen = set()
    for shape in range(mock_env._GLYPH_SHAPES):
        img = mock_env._glyph_icon(shape).pixmap(16, 16).toImage()
        seen.add(bytes(img.constBits()))
    assert len(seen) == mock_env._GLYPH_SHAPES, "有形状画出来是一模一样的"


def test_icon_check_flags_a_collapsed_unnamed_ratio():
    """给按钮加图标后若 UIA 开始报名字，10% 无名这条分布对齐会静默失效——要能报出来。"""
    import mock_v3

    spec = {"buttons": [("保存", True), ("", True)]}
    assert mock_v3._icon_controls_lost_their_blank_name(
        spec, [{"name": "保存"}, {"name": "确定"}]) is True
    assert mock_v3._icon_controls_lost_their_blank_name(
        spec, [{"name": "保存"}, {"name": ""}]) is False
    # spec 里本来就没有无名按钮 → 不检查，免得误报
    assert mock_v3._icon_controls_lost_their_blank_name(
        {"buttons": [("保存", True)]}, [{"name": "保存"}]) is False
