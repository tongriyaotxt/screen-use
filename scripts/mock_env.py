"""desktop-jev 腿 2：mock 应用合成器 v2（桌面版 MiniWoB++）。

v2 两处根本性改进（2026-09-18 owner 反馈）：
1. **零干扰**：UIA InvokePattern 点击（不动鼠标、不需前台）+ WA_ShowWithoutActivating
   窗口（不抢焦点、放角落）——量产时用户可正常使用电脑。
2. **语义+顺序，非位置**：确定/取消、保存/另存为等语义对抗同屏；状态依赖顺序
   （先选中才能删除）；向导/搜索/下拉/Tab/弹窗多步流程；同一意图多种表述；中英双语。

依赖：pip install pyside6（仅本脚本需要，screen_use 包不背 Qt）

用法（项目根，.venv）：
    ./.venv/Scripts/python.exe scripts/mock_env.py --episodes 500 --seed 42

输出：~/.screen_use/traces_mock/{episode_id}.jsonl（quality=gold, source=mock_oracle）
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLineEdit, QListWidget, QListWidgetItem,
    QCheckBox, QRadioButton, QComboBox, QTabWidget, QLabel, QStackedWidget,
    QMessageBox,
)

from screen_use.memory import _default_home
from screen_use.tools import ScreenUse

# ==================== 语料池 ====================

BTN_ACTIONS = ["提交", "确定", "保存", "搜索", "下一步", "完成", "导出", "刷新", "发送", "登录"]
BTN_CANCEL = ["取消", "返回", "放弃", "关闭"]
BTN_NEAR = {"保存": "另存为", "删除": "清除", "提交": "提交审核", "确定": "取消"}
FIELD_POOL = ["姓名", "城市", "电话", "邮箱", "地址", "备注", "公司", "邮编",
              "部门", "职位", "网址", "账号", "关键词", "用户名", "密码"]
VALUE_POOL = ["张三", "李四", "王五", "北京", "上海", "广州", "13800138000",
              "test@example.com", "朝阳区建国路100号", "100000", "技术部",
              "高级工程师", "admin001", "qwerty123"]
ITEM_POOL = ["苹果", "香蕉", "橘子", "葡萄", "西瓜", "项目Alpha", "项目Beta",
             "报告2026", "会议纪要", "待办事项", "草稿v1", "终稿v2"]
CHECK_POOL = ["记住我", "接收通知", "同意协议", "自动保存", "启用加速"]
RADIO_POOL = ["标准配送", "快速配送", "门店自提"]
COMBO_POOL = ["北京", "上海", "广州", "深圳", "杭州"]

EN = {  # 英文 UI 变体（部分 episode）
    "提交": "Submit", "确定": "OK", "保存": "Save", "搜索": "Search",
    "取消": "Cancel", "删除": "Delete", "下一步": "Next", "完成": "Finish",
    "姓名输入框": "Name input", "城市输入框": "City input",
}


def _tr(rng, text: str, english: bool) -> str:
    return EN.get(text, text) if english else text


# ==================== 窗口构建 ====================

class MockWindow(QWidget):
    """按 spec 构建的随机窗口。不抢焦点、固定角落小窗（零干扰）。"""

    def __init__(self, rng: random.Random, episode_id: int, spec: dict):
        super().__init__()
        self.rng = rng
        self.spec = spec
        self.setWindowTitle(f"MockUI-{episode_id}")
        # 零干扰：显示不激活、固定右下角小窗（UIA Invoke 点击本来就不会激活窗口，
        # 保留键盘焦点能力，否则下拉弹层/菜单弹层会因无焦点瞬关）。
        # 注意 Qt 坐标是逻辑像素（本机 1280×800），放物理坐标会把窗口甩出屏幕外。
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setGeometry(rng.randint(820, 1000), rng.randint(450, 620), 460, 340)

        self.widgets: dict[str, QWidget] = {}   # name -> widget（按 UIA 可见名索引）
        self.buttons: dict[str, QPushButton] = {}
        self.fields: dict[str, QLineEdit] = {}
        self.listbox: QListWidget | None = None
        self.list_items: list[str] = []
        self._build()

    def _build(self):
        rng, spec = self.rng, self.spec
        y = 30
        for name in spec.get("fields", []):
            e = QLineEdit(self)
            e.setAccessibleName(name)
            e.setGeometry(rng.randint(20, 60), y, rng.randint(160, 240), 30)
            if name in spec.get("readonly_fields", []):
                e.setReadOnly(True)
            self.fields[name] = e
            self.widgets[name] = e
            y += rng.randint(40, 52)

        if spec.get("combo"):
            c = QComboBox(self)
            c.addItems(spec["combo"]["items"])
            c.setAccessibleName(spec["combo"]["name"])
            c.setGeometry(rng.randint(250, 300), 30, 130, 30)
            self.widgets[spec["combo"]["name"]] = c
            self.combo = c
        else:
            self.combo = None

        for name in spec.get("checks", []):
            cb = QCheckBox(name, self)
            cb.setGeometry(rng.randint(20, 60), y, 150, 26)
            self.widgets[name] = cb
            y += 32

        if spec.get("radios"):
            for i, name in enumerate(spec["radios"]["items"]):
                rb = QRadioButton(name, self)
                rb.setGeometry(250 + (i % 2) * 100, y + (i // 2) * 30, 100, 26)
                self.widgets[name] = rb
            y += 34 * ((len(spec["radios"]["items"]) + 1) // 2)

        if spec.get("list"):
            items = list(spec["list"]["items"])  # 模板显式指定（oracle 必须知道内容）
            lb = QListWidget(self)
            lb.addItems(items)
            lb.setAccessibleName("项目列表")
            lb.setGeometry(280, 90, 150, 180)
            self.listbox = lb
            self.list_items = items

        # 按钮（含禁用态与联动逻辑）
        for i, (name, enabled) in enumerate(spec.get("buttons", [])):
            b = QPushButton(name, self)
            b.setGeometry(20 + (i % 3) * 140, 280 + (i // 3) * 45, 120, 34)
            b.setEnabled(enabled)
            self.buttons[name] = b
            self.widgets[name] = b

        # 联动：选中列表项 → 启用指定按钮（状态依赖顺序模板用）
        if spec.get("select_enables") and self.listbox is not None:
            targets = [self.buttons[n] for n in spec["select_enables"] if n in self.buttons]
            def on_select():
                for t in targets:
                    t.setEnabled(True)
            self.listbox.itemSelectionChanged.connect(on_select)

        # 联动：必填非空 → 启用提交（表单校验模板用）
        if spec.get("required_enables"):
            req_name, btn_name = spec["required_enables"]
            if req_name in self.fields and btn_name in self.buttons:
                btn = self.buttons[btn_name]
                def on_text(text):
                    btn.setEnabled(bool(text.strip()))
                self.fields[req_name].textChanged.connect(on_text)

        # 菜单栏（菜单导航范式）
        self.menus: dict = {}
        self.menubar = None
        if spec.get("menu"):
            from PySide6.QtWidgets import QMenuBar
            bar = QMenuBar(self)
            bar.setGeometry(0, 0, 460, 28)
            self.menubar = bar
            for menu_name, items in spec["menu"].items():
                m = bar.addMenu(menu_name)
                self.menus[menu_name] = m
                for it in items:
                    m.addAction(it)

        # 树形导航
        if spec.get("tree"):
            from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem
            tw = QTreeWidget(self)
            tw.setHeaderHidden(True)
            tw.setAccessibleName("导航树")
            tw.setGeometry(20, 90, 180, 170)
            for parent, children in spec["tree"].items():
                p = QTreeWidgetItem([parent])
                for c in children:
                    p.addChild(QTreeWidgetItem([c]))
                tw.addTopLevelItem(p)

        # 表格
        if spec.get("table"):
            from PySide6.QtWidgets import QTableWidget, QTableWidgetItem
            rows = spec["table"]
            tb = QTableWidget(len(rows), len(rows[0]), self)
            tb.setAccessibleName("数据表格")
            tb.setGeometry(20, 90, 400, 180)
            for r, row in enumerate(rows):
                for c, val in enumerate(row):
                    tb.setItem(r, c, QTableWidgetItem(val))

        # 模态弹窗联动：点击指定按钮弹出确认框（对话框栈范式）
        if spec.get("dialog_on_click"):
            trigger, dialog_btns = spec["dialog_on_click"]
            if trigger in self.buttons:
                def show_dialog():
                    from PySide6.QtWidgets import QDialog, QLabel
                    dlg = QDialog(self)
                    dlg.setWindowTitle(self.windowTitle())  # 同标题 → 观测按 tag 捕获
                    dlg.setGeometry(self.x() + 60, self.y() + 80, 300, 120)
                    QLabel("确定要执行该操作吗？", dlg).setGeometry(20, 15, 260, 30)
                    for i, bn in enumerate(dialog_btns):
                        b = QPushButton(bn, dlg)
                        b.setGeometry(40 + i * 130, 65, 100, 34)
                        b.clicked.connect(dlg.close)
                    dlg.setAttribute(Qt.WA_ShowWithoutActivating, True)
                    dlg.show()
                self.buttons[trigger].clicked.connect(show_dialog)


# ==================== 任务模板（语义+顺序） ====================
# 返回 (spec, goal, steps)；step = (action, target_name|None, text|None)
# action: click_id / type / scroll / wait / done / blocked

def _base_spec(rng, english=False):
    """每个 episode 的底噪：随机干扰按钮/输入框（位置随机，语义无关）。"""
    noise_btns = rng.sample(BTN_ACTIONS + BTN_CANCEL, rng.randint(1, 3))
    return {
        "fields": [f"{f}输入框" for f in rng.sample(FIELD_POOL, rng.randint(1, 3))],
        "buttons": [(b, True) for b in noise_btns],
        "checks": [], "radios": None, "combo": None, "list": None,
        "readonly_fields": [], "select_enables": [], "required_enables": None,
    }


def _add_btn(spec, name, enabled=True):
    if name not in [n for n, _ in spec["buttons"]]:
        spec["buttons"].append((name, enabled))


def t_semantic_choice(rng):
    """语义对抗：确定/取消（或保存/另存为）同屏，goal 表述决定选哪个。"""
    spec = _base_spec(rng)
    intent, positives, negatives = rng.choice([
        ("保存修改", ["保存", "确定"], ["取消", "另存为"]),
        ("放弃修改", ["取消"], ["确定", "保存"]),
        ("取消操作", ["取消"], ["确定", "提交"]),
    ])
    _add_btn(spec, rng.choice(positives))
    for neg in negatives:
        _add_btn(spec, neg)
    rng.shuffle(spec["buttons"])
    target = [n for n, _ in spec["buttons"] if n in positives][0]
    phrasings = {
        "保存修改": ["保存刚才的修改", "把改动保存下来", "确认并保存", "我要保留这些修改"],
        "放弃修改": ["放弃刚才的修改", "不要保存，直接取消", "这些改动不要了"],
        "取消操作": ["取消当前操作", "算了，不做这个操作了", "取消吧"],
    }
    return spec, rng.choice(phrasings[intent]), [
        ("click_id", target, None), ("done", None, "已完成")]


def t_state_dependent_delete(rng):
    """状态依赖顺序：删除初始禁用，选中列表项后才可用。"""
    spec = _base_spec(rng)
    items = rng.sample(ITEM_POOL, rng.randint(5, 8))
    spec["list"] = {"items": items}
    _add_btn(spec, "删除", enabled=False)
    spec["select_enables"] = ["删除"]
    item = rng.choice(items)
    goal = rng.choice([f"删除列表中的「{item}」", f"把「{item}」从列表里删掉", f"移除「{item}」"])
    return spec, goal, [
        ("click_id", item, None), ("click_id", "删除", None), ("done", None, "已删除")]


def t_form_required(rng):
    """表单校验顺序：必填非空才可用提交。"""
    spec = _base_spec(rng)
    req = spec["fields"][0]
    _add_btn(spec, "提交", enabled=False)
    spec["required_enables"] = (req, "提交")
    value = rng.choice(VALUE_POOL)
    goal = rng.choice([
        f"在「{req}」中填写「{value}」并提交",
        f"把「{value}」填到「{req}」里，然后提交表单",
    ])
    return spec, goal, [
        ("click_id", req, None), ("type", req, value),
        ("click_id", "提交", None), ("done", None, "表单已提交")]


def t_search_flow(rng):
    """搜索流：输入关键词 → 点搜索 → 点结果（三步顺序）。"""
    spec = _base_spec(rng)
    spec["fields"] = ["搜索框"]
    _add_btn(spec, "搜索")
    items = rng.sample(ITEM_POOL, rng.randint(5, 8))
    spec["list"] = {"items": items}
    item = rng.choice(items)
    goal = rng.choice([
        f"搜索「{item}」并在结果中选中它",
        f"在搜索框输入「{item}」，点搜索，然后点击搜索结果里的「{item}」",
    ])
    return spec, goal, [
        ("click_id", "搜索框", None), ("type", "搜索框", item),
        ("click_id", "搜索", None), ("click_id", item, None), ("done", None, "已找到并选择")]


def t_combo_select(rng):
    """下拉框两步：点开 → 选项。"""
    spec = _base_spec(rng)
    items = rng.sample(COMBO_POOL, 4)
    spec["combo"] = {"name": "城市选择", "items": items}
    item = rng.choice(items)
    goal = rng.choice([
        f"在下拉框中选择「{item}」", f"把城市改成「{item}」",
    ])
    return spec, goal, [
        ("click_id", "城市选择", None), ("click_id", item, None), ("done", None, "已选择")]


def t_checkbox(rng):
    """勾选/取消勾选。"""
    spec = _base_spec(rng)
    checks = rng.sample(CHECK_POOL, 3)
    spec["checks"] = checks
    target = rng.choice(checks)
    goal = rng.choice([f"勾选「{target}」", f"把「{target}」打上勾", f"启用「{target}」"])
    return spec, goal, [("click_id", target, None), ("done", None, "已勾选")]


def t_radio(rng):
    """单选。"""
    spec = _base_spec(rng)
    items = rng.sample(RADIO_POOL, 3)
    spec["radios"] = {"items": items}
    target = rng.choice(items)
    goal = rng.choice([f"选择「{target}」", f"配送方式选「{target}」"])
    return spec, goal, [("click_id", target, None), ("done", None, "已选择")]


def t_wizard(rng):
    """向导流：下一步 ×2 → 完成（多步导航顺序）。"""
    spec = _base_spec(rng)
    spec["buttons"] = [("下一步", True), ("上一步", True), ("完成", False)]
    # 简化：不真的换页，顺序考察为主
    goal = rng.choice([
        "完成设置向导（一路下一步直到完成）",
        "按向导流程走：连续点击下一步，最后点完成",
    ])
    return spec, goal, [
        ("click_id", "下一步", None), ("click_id", "下一步", None),
        ("click_id", "完成", None), ("done", None, "向导完成")]


def t_near_name(rng):
    """近名干扰：保存 vs 另存为、删除 vs 清除，必须精确匹配语义。"""
    spec = _base_spec(rng)
    pair = rng.choice(list(BTN_NEAR.items()))
    _add_btn(spec, pair[0])
    _add_btn(spec, pair[1])
    rng.shuffle(spec["buttons"])
    goal = rng.choice([
        f"点击「{pair[0]}」按钮（注意不是「{pair[1]}」）",
        f"点「{pair[0]}」，别点成「{pair[1]}」",
    ])
    return spec, goal, [("click_id", pair[0], None), ("done", None, "已点击")]


def t_blocked_missing(rng):
    """认输：目标元素不存在。"""
    spec = _base_spec(rng)
    present = [n for n, _ in spec["buttons"]]
    ghost = rng.choice([b for b in BTN_ACTIONS + BTN_CANCEL if b not in present])
    goal = f"点击「{ghost}」按钮"
    return spec, goal, [("blocked", None, f"元素不存在: {ghost}")]


def t_blocked_disabled(rng):
    """认输：目标存在但禁用，且无可行的启用路径。"""
    spec = _base_spec(rng)
    _add_btn(spec, "导出", enabled=False)
    goal = "点击「导出」按钮"
    return spec, goal, [("blocked", None, "元素禁用且无法启用: 导出")]


def t_scroll_list(rng):
    """滚动到底选择（Qt 虚拟化 → 先滚后点）。"""
    spec = _base_spec(rng)
    n = rng.randint(18, 26)
    items = rng.sample(ITEM_POOL, min(len(ITEM_POOL), n))
    if n > len(ITEM_POOL):
        items += [f"条目{i}" for i in range(n - len(ITEM_POOL))]
    spec["list"] = {"items": items}
    return spec, "滚动列表到底部并选择最后一项", [
        ("scroll", "项目列表", -8), ("scroll", "项目列表", -8),
        ("click_id", "__LAST_ITEM__", None), ("done", None, "已选择")]


def t_menu_nav(rng):
    """菜单导航：点开菜单 → 点菜单项（弹层两步）。"""
    spec = _base_spec(rng)
    spec["menu"] = {"文件": ["保存", "另存为", "退出"],
                    "编辑": ["撤销", "重做"],
                    "帮助": ["关于"]}
    menu, item = rng.choice([("文件", "保存"), ("文件", "另存为"), ("编辑", "撤销")])
    goal = rng.choice([
        f"打开「{menu}」菜单并点击「{item}」",
        f"在菜单栏的「{menu}」里点「{item}」",
    ])
    return spec, goal, [
        ("click_id", menu, None), ("click_id", item, None), ("done", None, "已执行菜单项")]


def t_tree_nav(rng):
    """树形导航：展开父节点 → 选子节点。"""
    spec = _base_spec(rng)
    tree = {"文档": ["报告", "笔记", "草稿"], "图片": ["截图", "照片"], "音乐": ["收藏"]}
    spec["tree"] = tree
    parent = rng.choice(["文档", "图片"])
    child = rng.choice(tree[parent])
    goal = rng.choice([
        f"在导航树中展开「{parent}」并选择「{child}」",
        f"找到树里的「{parent}」展开它，然后点「{child}」",
    ])
    return spec, goal, [
        ("click_id", parent, None), ("click_id", child, None), ("done", None, "已选择")]


def t_table_select(rng):
    """表格：按内容定位单元格。"""
    spec = _base_spec(rng)
    names = rng.sample(VALUE_POOL[:6], 3)
    rows = [[n, rng.choice(["技术部", "市场部"]), str(rng.randint(1000, 9999))] for n in names]
    spec["table"] = rows
    target = rng.choice(names)
    goal = rng.choice([f"在表格中点击「{target}」", f"选中表格里名字是「{target}」的单元格"])
    return spec, goal, [("click_id", target, None), ("done", None, "已选中")]


def t_dialog_confirm(rng):
    """模态弹窗：点删除 → 弹确认框 → 点确定/取消（对话框栈）。"""
    spec = _base_spec(rng)
    confirm = rng.random() < 0.5
    spec["dialog_on_click"] = ("删除", ["确定", "取消"])
    spec["buttons"] = [("删除", True)] + [(b, e) for b, e in spec["buttons"] if b != "删除"]
    if confirm:
        goal = rng.choice([
            "点击「删除」按钮，并在弹出的确认框中点击「确定」",
            "删除当前项目，弹出确认对话框时点「确定」",
        ])
        steps = [("click_id", "删除", None), ("click_id", "确定", None), ("done", None, "已确认删除")]
    else:
        goal = rng.choice([
            "点击「删除」按钮，但在确认框中点「取消」放弃删除",
            "点「删除」后反悔：在确认框里点「取消」",
        ])
        steps = [("click_id", "删除", None), ("click_id", "取消", None), ("done", None, "已取消删除")]
    return spec, goal, steps


TEMPLATES = [
    (t_semantic_choice, 3), (t_state_dependent_delete, 2), (t_form_required, 2),
    (t_search_flow, 2), (t_combo_select, 1), (t_checkbox, 1), (t_radio, 1),
    (t_wizard, 1), (t_near_name, 2), (t_blocked_missing, 1), (t_blocked_disabled, 1),
    (t_scroll_list, 1),
    (t_menu_nav, 2), (t_tree_nav, 2), (t_table_select, 1), (t_dialog_confirm, 2),
]
_WEIGHTED = [t for t, w in TEMPLATES for _ in range(w)]


# ==================== 执行（零鼠标） ====================

class EpisodeRunner:
    def __init__(self, out_dir: Path):
        self.tools = ScreenUse()
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    def _observe(self, episode_id: int) -> list[dict]:
        """观测 = mock 窗口元素 + 落在窗口矩形内的全局元素（下拉弹窗/浮层）。

        下拉框弹出层是独立窗口（窗口矩形近似由 mock 元素的 bbox 并集给出，
        下缘留 margin 覆盖向下弹出的列表）——与 VLM 实际看到的画面一致。"""
        data = self.tools.list_ui_elements()
        tag = f"MockUI-{episode_id}"
        own = [e for e in data["elements"] if tag in e["window_title"]]
        if own:
            left = min(e["bbox"][0] for e in own) - 50
            top = min(e["bbox"][1] for e in own) - 50
            right = max(e["bbox"][2] for e in own) + 50
            bottom = max(e["bbox"][3] for e in own) + 250  # 下拉向下弹出
            extra = [e for e in data["elements"]
                     if tag not in e["window_title"]
                     and left <= e["center"][0] <= right
                     and top <= e["center"][1] <= bottom]
            own.extend(extra)
        for i, e in enumerate(own, 1):
            e["_gid"] = e["id"]
            e["id"] = i
        return own

    @staticmethod
    def _find(elements, name):
        for e in elements:
            if e["name"] == name:
                return e
        return None

    def _raw_control(self, gid: int):
        for el in self.tools._elements:  # 全局枚举缓存（带 control 对象）
            if el.id == gid:
                return el.control
        return None

    def _invoke(self, gid: int) -> bool:
        """UIA 模式点击：不动鼠标、不需前台（零干扰核心）。

        按控件语义选模式：EditControl→SetFocus（点输入框=获得焦点）；
        ComboBox→ExpandCollapse.Expand；按钮→Invoke；列表项→SelectionItem.Select；
        勾选/单选→Toggle。"""
        ctrl = self._raw_control(gid)
        if ctrl is None:
            return False
        try:
            if ctrl.ControlTypeName == "EditControl":
                ctrl.SetFocus()
                return True
        except Exception:
            pass
        # 树节点/下拉框优先展开（Expand 才是"点击"的语义），其余按模式尝试
        if ctrl.ControlTypeName in ("TreeItemControl", "ComboBoxControl"):
            try:
                ctrl.GetExpandCollapsePattern().Expand()
                return True
            except Exception:
                pass
        attempts = [
            ("GetInvokePattern", lambda p: p.Invoke()),
            ("GetSelectionItemPattern", lambda p: p.Select()),
            ("GetTogglePattern", lambda p: p.Toggle()),
            ("GetExpandCollapsePattern", lambda p: p.Expand()),
        ]
        for getter, act in attempts:
            try:
                act(getattr(ctrl, getter)())
                return True
            except Exception:
                continue
        return False

    def _close_popups(self, win: MockWindow) -> None:
        """强制关闭弹层（防 grabMouse 劫持用户鼠标——2026-09-19 事故）。"""
        try:
            if win.combo is not None and win.combo.view().isVisible():
                win.combo.hidePopup()
        except Exception:
            pass
        for m in getattr(win, "menus", {}).values():
            try:
                if m.isVisible():
                    m.close()
            except Exception:
                pass

    def _click_target(self, win: MockWindow, target: str, el: dict) -> bool:
        """点击目标：菜单弹层走 Qt 直驱（探针证实 popup() 可持久）；下拉框走 UIA
        ExpandCollapse（Qt showPopup 对无激活窗口瞬关，UIA Expand 反而稳定）。"""
        if target in win.menus and win.menubar is not None:
            menu = win.menus[target]
            rect = win.menubar.actionGeometry(menu.menuAction())
            menu.popup(win.menubar.mapToGlobal(rect.bottomLeft()))
            return True
        return self._invoke(el["_gid"])

    def run(self, win: MockWindow, goal: str, steps: list, episode_id: str) -> dict:
        path = self.out_dir / f"{episode_id}.jsonl"
        records = []
        stats = {"steps": 0, "ok": 0}
        for idx, (action, target, text) in enumerate(steps, 1):
            if target == "__LAST_ITEM__":
                target = win.list_items[-1]
            elements = self._observe(episode_id)
            title = f"MockUI-{episode_id}"
            args: dict = {}
            ok, detail = True, ""
            el = self._find(elements, target) if target else None

            if action == "click_id":
                if el is None:
                    ok, detail = False, f"oracle 目标不可见: {target}"
                else:
                    args = {"id": el["id"]}
                    ok = self._click_target(win, target, el)
                    detail = f"点击 {target}" + ("" if ok else "（invoke 失败）")
            elif action == "type":
                if el is None:
                    ok, detail = False, f"oracle 目标不可见: {target}"
                else:
                    args = {"text": text, "element_id": el["id"]}
                    r = self.tools.set_element_text(el["_gid"], text)
                    ok = bool(r.get("ok"))
                    detail = f"输入 {text!r} → {target}"
            elif action == "scroll":
                args = {"clicks": text}
                # 零鼠标：Qt 滚动条直接驱动（我们拥有窗口对象）
                if win.listbox is not None:
                    sb = win.listbox.verticalScrollBar()
                    sb.setValue(min(sb.maximum(), sb.value() + 100 * abs(int(text))))
                detail = f"滚动 {text}"
            elif action == "blocked":
                args = {"reason": text}
                detail = text
            elif action == "done":
                args = {"result": text}
                detail = "任务完成"

            records.append({
                "goal": goal, "step_idx": idx, "foreground_title": title,
                "elements": [{k: v for k, v in e.items() if k != "_gid"} for e in elements],
                "action": action, "args": args, "ok": ok, "detail": detail,
                "ts": time.time(), "source": "mock_oracle", "quality": "gold",
            })
            stats["steps"] += 1
            stats["ok"] += ok
            QApplication.processEvents()
            time.sleep(0.15)
            # 弹层生命周期管理：只有"刚打开弹层"的步之后才允许弹层存活到下一步观测；
            # 其余情况立即关闭（点击失败也必须关，防 grabMouse 劫持用户鼠标）
            opened_popup = (action == "click_id" and ok and (
                target in getattr(win, "menus", {})
                or (win.combo is not None
                    and win.spec.get("combo", {}).get("name") == target)))
            if not opened_popup:
                self._close_popups(win)
        self._close_popups(win)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return stats


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=1,
                        help="从第 N 个 episode 开始（rng 序列一致，断点续跑）")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    app = QApplication(sys.argv)
    out_dir = _default_home() / "traces_mock"
    runner = EpisodeRunner(out_dir)
    total = {"episodes": 0, "steps": 0, "ok": 0}

    for ep in range(1, args.episodes + 1):
        episode_id = f"mock-v2-{args.seed}-{ep:04d}"
        template = _WEIGHTED[(ep - 1) % len(_WEIGHTED)]
        spec, goal, steps = template(rng)
        if ep < args.start:
            continue  # 保持 rng 消费一致，跳过已完成的 episode
        win = MockWindow(rng, episode_id, spec)
        win.show()
        app.processEvents()
        time.sleep(0.5)
        stats = runner.run(win, goal, steps, episode_id)
        win.close()
        app.processEvents()
        total["episodes"] += 1
        total["steps"] += stats["steps"]
        total["ok"] += stats["ok"]
        if ep % 25 == 0:
            print(f"[{ep}/{args.episodes}] steps={total['steps']} ok={total['ok']}", flush=True)

    print(f"完成: {total['episodes']} episodes, {total['steps']} 步 "
          f"(oracle 成功 {total['ok']}) → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
