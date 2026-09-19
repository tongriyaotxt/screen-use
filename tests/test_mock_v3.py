"""mock v3 生成器测试：尺度、双语、语义措辞、oracle 标签有效性、真 Qt 布局体检。"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import mock_v3  # noqa: E402


def spec_names(spec: dict) -> set[str]:
    """窗口里所有可被 UIA 看到的控件名（含弹窗按钮）。"""
    names: set[str] = set(spec.get("fields", []))
    names |= {n for n, _ in spec.get("buttons", [])}
    names |= set(spec.get("checks", []))
    if spec.get("radios"):
        names |= set(spec["radios"]["items"])
    if spec.get("combo"):
        names.add(spec["combo"]["name"])
        names |= set(spec["combo"]["items"])
    if spec.get("list"):
        names |= set(spec["list"]["items"])
    for parent, children in (spec.get("tree") or {}).items():
        names.add(parent)
        names |= set(children)
    for row in spec.get("table") or []:
        names |= set(row)
    for menu, items in (spec.get("menu") or {}).items():
        names.add(menu)
        names |= set(items)
    if spec.get("dialog_on_click"):
        trigger, btns = spec["dialog_on_click"]
        names.add(trigger)
        names |= set(btns)
    return names


def oracle_targets(steps: list) -> list[str]:
    return [t for a, t, _ in steps
            if a in ("click_id", "type") and t and t != "__LAST_ITEM__"]


# ---- 尺度：对齐真实桌面（元素数中位 84，v2 只有 14） ----

def test_densify_reaches_realistic_scale():
    rng = random.Random(0)
    spec, _, _ = mock_v3.build_episode(rng, 1, target_elements=70, bilingual_prob=0,
                                       semantic_prob=0)
    assert mock_v3.estimate_elements(spec) >= 40      # v2 的 14 → 至少 40
    assert spec["size"][0] >= 800                     # 大窗口才装得下


def test_densify_big_profile_has_tree_table_menu():
    rng = random.Random(1)
    spec, _, _ = mock_v3.build_episode(rng, 2, target_elements=100, bilingual_prob=0,
                                       semantic_prob=0)
    assert spec.get("tree") and spec.get("table") and spec.get("menu")
    assert mock_v3.estimate_elements(spec) >= 70


def test_densify_hits_target_scale():
    """目标 85（真实中位 84）时，产出应落在 80–115 之间，不能失控。"""
    rng = random.Random(30)
    counts = [mock_v3.estimate_elements(
        mock_v3.build_episode(rng, ep, target_elements=85)[0]) for ep in range(1, 31)]
    assert min(counts) >= 80
    assert max(counts) <= 115


def test_densify_keeps_small_profile_available():
    """target_elements 小 → 不塞干扰（保留 v2 那样的干净窗口，供对照）。"""
    rng = random.Random(2)
    spec = {"fields": ["f"], "buttons": [("b", True)], "checks": [], "tree": None,
            "table": None, "menu": None, "size": (460, 340)}
    out = mock_v3.densify(spec, rng, target_elements=20)
    assert not out.get("tree") and not out.get("table") and not out.get("menu")
    assert out["size"] == (460, 340)
    assert len(out["buttons"]) == 1                    # 不加干扰按钮


def test_densify_never_overwrites_template_panels():
    """模板自带的树/表/菜单里可能有 oracle 目标 —— densify 只能合并/追加，不能覆盖。"""
    rng = random.Random(11)
    spec = {"fields": [], "buttons": [], "checks": [],
            "tree": {"文档": ["报告"]}, "menu": {"文件": ["保存"]}, "table": None}
    out = mock_v3.densify(spec, rng, target_elements=100)
    assert "报告" in out["tree"]["文档"]                # 模板的树节点还在
    assert "保存" in out["menu"]["文件"]                # 模板的菜单项还在


def test_densify_caps_fields_to_avoid_overlap():
    """输入框从 field_y0 往下排，太多会压住按钮区（布局测试会挂）——densify 自己封顶。"""
    rng = random.Random(12)
    spec = {"fields": ["a", "b", "c"], "buttons": [("x", True)], "checks": []}
    out = mock_v3.densify(spec, rng, target_elements=100)
    assert len(out["fields"]) <= 3


# ---- oracle 标签有效性（最关键：量产出的轨迹必须条条可执行） ----

@pytest.mark.parametrize("seed", range(6))
def test_oracle_targets_always_exist_in_spec(seed):
    rng = random.Random(seed)
    for ep in range(1, 26):
        spec, goal, steps = mock_v3.build_episode(rng, ep, target_elements=85)
        names = spec_names(spec)
        missing = [t for t in oracle_targets(steps) if t not in names]
        assert not missing, f"seed={seed} ep={ep} goal={goal!r} 缺失目标 {missing}"


# ---- 双语命名：中文目标 ↔ 英文控件 ----

def test_bilingual_renames_and_keeps_steps_in_sync():
    rng = random.Random(3)
    spec = {"fields": ["姓名输入框"], "checks": [], "buttons": [("保存", True), ("取消", True)]}
    steps = [("click_id", "保存", None)]
    new_spec, new_steps = mock_v3.bilingual(spec, steps, rng, prob=1.0)
    assert ("Save", True) in new_spec["buttons"]
    assert new_steps[0][1] == "Save"                  # 步骤跟着改，oracle 仍可执行
    assert "Save" in spec_names(new_spec)


def test_bilingual_protects_data_names():
    """列表项/菜单项等数据名不改（goal 里引用着它们，改了就对不上）。"""
    rng = random.Random(4)
    spec = {"fields": [], "checks": [], "buttons": [],
            "list": {"items": ["保存", "确定"]}}
    new_spec, _ = mock_v3.bilingual(spec, [], rng, prob=1.0)
    assert new_spec["list"]["items"] == ["保存", "确定"]


def test_bilingual_can_be_disabled():
    rng = random.Random(5)
    spec = {"fields": [], "checks": [], "buttons": [("保存", True)]}
    new_spec, _ = mock_v3.bilingual(spec, [("click_id", "保存", None)], rng, prob=0.0)
    assert ("保存", True) in new_spec["buttons"]


def test_bilingual_never_creates_duplicate_names():
    """改名不能造出重名控件：干扰池里本来就有 "Save"/"Edit"，撞名会让 oracle 找错目标。"""
    rng = random.Random(40)
    spec = {"fields": [], "checks": [], "buttons": [("保存", True), ("Save", True)]}
    steps = [("click_id", "保存", None)]
    for seed in range(20):
        new_spec, _ = mock_v3.bilingual(spec, steps, random.Random(seed), prob=1.0)
        names = [n for n, _ in new_spec["buttons"]]
        assert len(names) == len(set(names)), f"seed={seed} 出现重名: {names}"


# ---- 语义措辞：goal 不再照抄控件名 ----

def test_semantic_goal_avoids_quoting_target_name():
    rng = random.Random(6)
    goal = "点击「保存」按钮（注意不是「另存为」）"
    new = mock_v3.semantic_goal(goal, [("click_id", "保存", None)], rng, prob=1.0)
    assert "保存" not in new
    assert new != goal


def test_semantic_goal_keeps_input_value_quoting():
    """要输入的文本值必须照实引用（goal-copy 机制靠它）——模板里的值不在改写范围内。"""
    rng = random.Random(7)
    goal = "在「姓名输入框」中填写「张三」并提交"
    new = mock_v3.semantic_goal(goal, [("click_id", "姓名输入框", None)], rng, prob=1.0)
    assert "张三" in new                              # 值仍原样保留


def test_semantic_goal_noop_without_phrase():
    rng = random.Random(8)
    goal = "删除列表中的「苹果」"
    assert mock_v3.semantic_goal(goal, [("click_id", "苹果", None)], rng, prob=1.0) == goal
    assert mock_v3.semantic_goal(goal, [("click_id", "苹果", None)], rng, prob=0.0) == goal


def test_semantic_goal_only_replaces_quoted_names():
    """控件名常常同时是动词（"删除列表中的「苹果」"）——只替换引号内的，别改坏句子。"""
    rng = random.Random(12)
    goal = "删除列表中的「苹果」"
    assert mock_v3.semantic_goal(goal, [("click_id", "苹果", None),
                                        ("click_id", "删除", None)], rng, prob=1.0) == goal


def test_semantic_goal_handles_later_semantic_target():
    """多步流程里第一个可点目标常是输入框，真正要改写的是后面那个语义按钮。"""
    rng = random.Random(13)
    goal = "在搜索框输入「报告」，点搜索，然后点击结果里的「报告」"
    steps = [("click_id", "搜索框", None), ("type", "搜索框", "报告"),
             ("click_id", "搜索", None)]
    new = mock_v3.semantic_goal(goal, steps, rng, prob=1.0)
    assert "「搜索」" not in new


def test_semantic_rate_reduces_name_quoting():
    """跑一批回合：引用式目标（「保存」这类）多数应已被改成语义表述。"""
    rng = random.Random(10)
    quoted = 0
    total = 0
    for ep in range(1, 61):
        _, goal, steps = mock_v3.build_episode(rng, ep, target_elements=85)
        targets = [t for a, t, _ in steps
                   if a == "click_id" and t in mock_v3.SEMANTIC_PHRASES]
        if not targets:
            continue
        total += 1
        if any(f"「{t}」" in goal for t in targets):
            quoted += 1
    assert total > 10
    assert quoted / total < 0.4          # v2 是 100% 照抄


# ---- 整体：确定性 + 元素计数 ----

def test_build_episode_is_deterministic():
    a = mock_v3.build_episode(random.Random(9), 5, target_elements=85)
    b = mock_v3.build_episode(random.Random(9), 5, target_elements=85)
    assert a == b


def test_estimate_elements_counts_all_widget_kinds():
    spec = {"fields": ["a", "b"], "buttons": [("x", True)], "checks": ["c"],
            "radios": {"items": ["r1", "r2"]}, "combo": {"name": "k", "items": ["i"]},
            "list": {"items": ["l1", "l2", "l3"]},
            "tree": {"p": ["c1", "c2"]}, "table": [["t1", "t2"]],
            "menu": {"m": ["mi1", "mi2"]}}
    # 2 字段 + 1 按钮 + 1 勾选 + 2 单选 + 1 下拉 + 3 列表 + 3 树
    # + 1 表格 + 2 单元格 + 1 菜单名 + 2 菜单项 = 19
    assert mock_v3.estimate_elements(spec) == 19


# ---- 尺度验收：v3 的观测要真的贴到真实桌面的量级 ----

def test_v3_observation_scale_matches_real_desktop():
    """元素数中位要落在真实桌面量级（84）附近；v2 只有 14。

    token 量级由 `mock_v3.py --estimate-only` 报（需 tokenizer，测试里不加载 transformers）。
    """
    rng = random.Random(31)
    specs = [mock_v3.build_episode(rng, ep, target_elements=85)[:2]
             for ep in range(1, 41)]
    report = mock_v3.scale_report(specs)          # 不传 ckpt → 只算元素数
    assert 75 <= report["elements_p50"] <= 100
    assert report["elements_min"] >= 70


def test_v3_unnamed_icon_ratio_matches_real():
    """真实桌面 11% 的元素无名（图标按钮，全是 ButtonControl）。

    v2 是 0%——模型会以为"每个控件都有名字"。v3 按同量级掺无名控件。
    """
    import statistics

    rng = random.Random(33)
    ratios = []
    for ep in range(1, 31):
        spec, _, _ = mock_v3.build_episode(rng, ep, target_elements=85)
        els = mock_v3.synthesize_elements(spec)
        ratios.append(sum(1 for e in els if not e["name"].strip()) / len(els))
    assert 0.06 <= statistics.median(ratios) <= 0.20, f"无名占比 {statistics.median(ratios):.0%}"


# ---- 数据通路验收：v3 形状的轨迹喂给训练侧不能出错 ----

def test_v3_traces_survive_training_data_path(tmp_path):
    """按 v3 的观测尺度造一条合成轨迹，走一遍 iter_samples：

    - 样本数 = 动作集内的步数
    - elem_slot 必须指向 gold 元素（不是 no-element 槽）
    - 元素数不能超过 MAX_ELEMENTS_PER_SAMPLE（超了标签会被静默截断）
    """
    import json
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from training.common import MAX_ELEMENTS_PER_SAMPLE, iter_samples

    rng = random.Random(32)
    spec, goal, steps = mock_v3.build_episode(rng, 1, target_elements=85)
    elements = mock_v3.synthesize_elements(spec)
    assert len(elements) <= MAX_ELEMENTS_PER_SAMPLE

    path = tmp_path / "mock-v3-test-0001.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for idx, (action, target, value) in enumerate(steps, 1):
            args = {}
            if action == "click_id":
                hit = next((e for e in elements if e["name"] == target), None)
                if hit is None:
                    continue
                args = {"id": hit["id"]}
            elif action == "type":
                hit = next((e for e in elements if e["name"] == target), None)
                args = {"text": value, "element_id": hit["id"] if hit else None}
            elif action == "blocked":
                args = {"reason": value}
            f.write(json.dumps({
                "goal": goal, "step_idx": idx, "foreground_title": "MockUI",
                "elements": elements, "action": action, "args": args,
                "ok": True, "detail": f"{action} {target}",
                "source": "mock_oracle", "quality": "gold", "ts": 0.0,
            }, ensure_ascii=False) + "\n")

    samples = iter_samples([tmp_path])
    assert samples, "iter_samples 没产出样本"
    for s in samples:
        assert s["n_elements"] == len(elements)
        if s["op"] in ("CLICK", "TYPE"):
            assert s["elem_slot"] < len(elements), "gold 元素槽位应当命中具体元素"
    assert samples[0]["goal"] == goal
    assert samples[0]["history"] == []             # 第一步没有历史


# ---- 真 Qt 布局体检（offscreen，不开窗） ----
# v3 给面板显式坐标、按钮自动落到输入区下方——算错就会叠控件或被判 offscreen，
# UIA 看不到 → oracle 量产那天集体失败。这里用 offscreen 平台把布局钉死。

def _layout_app(monkeypatch):
    import os

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _all_widgets(win):
    """窗口里所有实际控件（面板 + 输入 + 按钮…）。

    不能用 win.widgets（按名字索引的 dict，重名控件——比如 v3 的无名按钮——会互相覆盖）。
    面板内部的行/单元格是 Qt model item 而非 QWidget，UIA 照样看得到它们
    （TreeItem/DataItem/ListItem），但布局由面板自己管，不在本测试范围。
    """
    from PySide6.QtWidgets import (QCheckBox, QComboBox, QLineEdit, QListWidget,
                                   QMenuBar, QPushButton, QRadioButton, QTableWidget,
                                   QTreeWidget)

    out = []
    for cls in (QPushButton, QLineEdit, QCheckBox, QRadioButton, QComboBox,
                QListWidget, QTableWidget, QTreeWidget, QMenuBar):
        out += [(cls.__name__, w) for w in win.findChildren(cls)]
    return out


def test_v3_layout_panels_do_not_overlap(monkeypatch):
    from PySide6.QtWidgets import (QComboBox, QListWidget, QMenuBar, QTableWidget,
                                   QTreeWidget)

    app = _layout_app(monkeypatch)
    import mock_env

    rng = random.Random(21)
    for ep in range(1, 21):
        spec, _, _ = mock_v3.build_episode(rng, ep, target_elements=85)
        win = mock_env.MockWindow(rng, f"layout-{ep}", spec)
        try:
            panels = []
            for cls in (QTreeWidget, QListWidget, QTableWidget, QComboBox, QMenuBar):
                panels += [(cls.__name__, w.geometry()) for w in win.findChildren(cls)]
            assert len(panels) >= 3, f"ep={ep} 面板没找到，测试等于没跑"
            for i in range(len(panels)):
                for j in range(i + 1, len(panels)):
                    a, b = panels[i][1], panels[j][1]
                    assert not a.intersects(b), (
                        f"ep={ep} 面板重叠: {panels[i][0]}{a.getRect()} "
                        f"vs {panels[j][0]}{b.getRect()}")
        finally:
            win.close()
        app.processEvents()


def test_v3_layout_widgets_stay_inside_window(monkeypatch):
    app = _layout_app(monkeypatch)
    import mock_env

    rng = random.Random(22)
    for ep in range(1, 21):
        spec, _, _ = mock_v3.build_episode(rng, ep, target_elements=85)
        w, h = spec["size"]
        win = mock_env.MockWindow(rng, f"bounds-{ep}", spec)
        try:
            widgets = _all_widgets(win)
            assert len(widgets) >= 8, f"ep={ep} 控件太少（{len(widgets)}）"
            for name, widget in widgets:
                g = widget.geometry()
                assert g.left() >= 0 and g.top() >= 0, (
                    f"ep={ep} {name} 越出左上 {g.getRect()}")
                assert g.right() <= w and g.bottom() <= h, (
                    f"ep={ep} {name} 越出窗口 {g.getRect()} > ({w},{h})")
        finally:
            win.close()
        app.processEvents()


def test_v3_layout_widgets_do_not_overlap(monkeypatch):
    """控件互相覆盖会被 Qt 判 offscreen → UIA 丢元素 → oracle 失败。"""
    app = _layout_app(monkeypatch)
    import mock_env

    rng = random.Random(23)
    for ep in range(1, 21):
        spec, _, _ = mock_v3.build_episode(rng, ep, target_elements=85)
        win = mock_env.MockWindow(rng, f"overlap-{ep}", spec)
        try:
            items = [(n, w.geometry()) for n, w in _all_widgets(win)]
            assert len(items) >= 8, f"ep={ep} 控件太少（{len(items)}）"
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a, b = items[i][1], items[j][1]
                    assert not a.intersects(b), (
                        f"ep={ep} 控件重叠: {items[i][0]}{a.getRect()} "
                        f"vs {items[j][0]}{b.getRect()}")
        finally:
            win.close()
        app.processEvents()
