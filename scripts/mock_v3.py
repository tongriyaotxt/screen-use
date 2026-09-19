"""mock v3：把合成观测的**尺度**与**语义**搬到真实分布上（依据 BENCH.md §7 的实测差距）。

v2 的合成数据有一个致命偏差：观测太干净。实测对比（2026-09-19）：

| | 元素数中位 | token 中位 | 目标措辞 |
|---|---|---|---|
| mock v2 训练集 | 14 | 457 | goal 直接照抄控件名（「保存」→ 按钮"保存"） |
| 真实桌面 | **84**（最大 107） | **2380** | goal 靠语义（"78 乘以 9" → 按钮 `Multiply by`） |

结果：同一个模型在 mock 上 joint_acc 0.77、真实域 0.17（升级率 90%）。v3 就补这三件事：

1. **densify**：往窗口里塞干扰控件（树/表格/菜单/工具条/无名按钮），把元素数抬到 40–120；
2. **bilingual**：一部分控件改用英文名（目标与干扰都可能），逼模型学"中文目标 ↔ 英文控件"，
   而不是字符串照抄；改名同步改 oracle 步骤，标签仍是 gold；
3. **semantic goal**：把"引用控件名"的目标改成同义/上下文表述（"把改动存下来" 而不是 "点击「保存」"）。
   数据类目标（列表项、表格单元格、要输入的文本值）仍照实引用——真实用户也是这么说话的。

用法（项目根，.venv；**会开窗，owner 离开电脑时跑**）：
    ./.venv/Scripts/python.exe scripts/mock_v3.py --episodes 2000 --seed 7
    ./.venv/Scripts/python.exe scripts/mock_v3.py --episodes 200 --seed 7 --target-elements 80
输出：~/.screen_use/traces_mock_v3/{episode_id}.jsonl（quality=gold, source=mock_oracle）
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mock_env  # noqa: E402  (复用窗口构建 / oracle 执行 / 弹层管理)
from screen_use.memory import _default_home  # noqa: E402

# ==================== 词池 ====================

# 干扰控件：真实应用里大量存在的、跟当前任务无关的东西。
# 2026-09-19 实测 traces_full：**76% 的可见元素是按钮**（67/步，其它类型加起来才 21）。
# 真实桌面上叠着计算器/终端这类按钮密集的窗口，所以按钮池必须够大——
# 而且按钮是唯一不受"视口虚拟化"限制的控件（网格排布，全都在视口里），
# 拿它当达标的主力填充料最稳。
DISTRACTOR_BUTTONS = [
    "File", "Edit", "View", "Help", "Refresh", "Export CSV", "Import", "Settings",
    "Profile", "Notifications", "Sync", "Filters", "Sort", "Columns", "Zoom In",
    "Zoom Out", "Fullscreen", "Share", "Print Preview", "Duplicate", "Archive",
    "Mark as Read", "Pin", "Collapse All", "Expand All",
    "Cut", "Copy", "Paste", "Undo", "Redo", "Delete", "Rename", "Move", "Add Row",
    "Add Column", "Export PDF", "Export Excel", "Publish", "Subscribe", "Like",
    "Comment", "Reply", "Forward", "Download", "Upload", "Backup", "Restore",
    "Run", "Debug", "Deploy", "Commit", "Push", "Pull", "Merge", "Terminal",
    "Extensions", "Format", "Insert", "Chart", "Link", "Footnote", "Header",
    "Footer", "Page Setup", "Spell Check", "Track Changes", "Reset", "Apply",
    "Discard", "Next", "Previous", "Finish", "Skip", "Retry", "Cancel All",
    "New Window", "Split View", "Reload", "Bookmark", "History", "DevTools",
]

# 图标按钮：真实桌面 **11% 的元素是无名的**（实测 traces_full，全是 ButtonControl，即 toolbar/
# 窗口按钮这类只有图标没有文字的控件），gold 目标里 0 个无名。v3 按同样比例掺无名控件，
# 否则模型会以为"每个控件都有名字"，遇到图标就抓瞎。
DISTRACTOR_ICONS_PER_EPISODE = 6
DISTRACTOR_FIELDS = [
    "Search", "Filter", "Quick find", "Address", "Comments", "Tags", "Reference",
    "Notes", "Region", "Timezone", "",
]
TREE_PARENTS = ["Workspace", "Documents", "Media", "Archive", "Shared", "Trash",
                "文档", "图片", "收藏"]
TREE_CHILDREN = ["Draft", "Final", "Notes", "Report", "Screenshot", "Invoice",
                 "草稿", "报告", "笔记", "截图", "发票"]
TABLE_COLS = ["Name", "Department", "Amount", "Status", "Owner", "Updated"]
TABLE_CELLS = ["Alice", "Bob", "Carol", "Dave", "Tech", "Sales", "Finance",
               "1000", "2400", "5800", "Open", "Closed", "Pending",
               "张三", "李四", "技术部", "市场部", "已审核"]
MENU_POOL = {
    "File": ["New", "Open", "Save As", "Export", "Print", "Close"],
    "Edit": ["Undo", "Redo", "Cut", "Copy", "Paste", "Find"],
    "View": ["Zoom", "Layout", "Sidebar", "Status Bar"],
    "Help": ["Docs", "Shortcuts", "About"],
}

# 语义化措辞：键 = v2 里会被照抄的控件名，值 = 不出现该名字的说法（真实用户就这么讲）
SEMANTIC_PHRASES = {
    "保存": ["把刚才的改动存下来", "让这份修改落盘", "留下这次的改动"],
    "确定": ["确认这次操作", "就按这个来", "认可当前设置"],
    "取消": ["放弃这次操作", "算了不做了", "撤销刚才那步"],
    "提交": ["把表单交上去", "送审这份表单"],
    "删除": ["把这一项去掉", "清理掉选中项", "移除它"],
    "搜索": ["查一下", "找找看", "检索一下"],
    "下一步": ["继续往下走", "进入下一环节"],
    "完成": ["收尾", "结束这个流程"],
    "关闭": ["把窗口收掉", "不看了"],
    "导出": ["把数据存成文件", "导出到本地"],
    "刷新": ["重新拉一次", "更新一下显示"],
}

# 英文替换池：把中文控件名换成英文（或反过来）——逼模型学语义而不是字符串
EN_RENAME = {
    "保存": "Save", "确定": "OK", "取消": "Cancel", "提交": "Submit",
    "删除": "Delete", "搜索": "Search", "下一步": "Next", "完成": "Finish",
    "关闭": "Close", "导出": "Export", "刷新": "Refresh", "重置": "Reset",
    "记住我": "Remember me", "接收通知": "Notifications", "同意协议": "Accept terms",
    "标准配送": "Standard shipping", "快速配送": "Express shipping",
    "门店自提": "Store pickup", "添加": "Add", "编辑": "Edit", "重命名": "Rename",
}


# ==================== 生成 ====================

def densify(spec: dict, rng: random.Random, target_elements: int) -> dict:
    """往 spec 里塞干扰控件，边加边算，直到预计元素数达到 target_elements。

    关键约束：**只加不覆盖**——模板自带的树/表/菜单里可能有 oracle 的目标
    （t_menu_nav 的「文件」、t_tree_nav 的「照片」），覆盖掉就等于把标签弄脏了。
    大窗口给各面板显式坐标，避免互相遮挡（Qt 遮挡会让 UIA 报 offscreen 而丢元素）。
    """
    spec = dict(spec)
    size = max(20, min(200, target_elements))
    big = size >= 70
    mid = size >= 40

    # 布局按「屏幕像素预算」排：元素数受可见面积限制（每个控件要占几十像素）。
    # 屏幕 1280×800：窗口 1200×740 @ y=20，底边 760 仍在屏内。
    # 竖直分区：菜单 0–28 / 面板 40–240 / 输入 260–400 / 勾选 400–460 / 按钮网格 460–740。
    # 按钮网格 9 列 × 8 行 = 72 个位置——真实桌面 76% 的可见元素是按钮，
    # 之前只留了 3 行（24 个）就再也加不动，才导致窗口永远到不了 85。
    if big or mid:
        spec.update({
            "size": (1200, 740), "menu_pos": (0, 0, 1200, 28),
            "tree_pos": (760, 40, 200, 200), "list_pos": (980, 40, 200, 200),
            "table_pos": (10, 40, 500, 200), "field_y0": 260,
            "combo_pos": (760, 260, 160, 30),
            "button_pos": (10, 460), "button_cols": 9,
            "button_size": (90, 26), "button_gap": (10, 8),
            "expand_tree": True,
        })

    def est() -> int:
        return estimate_elements(spec)

    def add_tree(n_parents: int, n_children: int) -> None:
        if spec.get("tree") or not mid:
            return
        parents = rng.sample(TREE_PARENTS, min(n_parents, len(TREE_PARENTS)))
        spec["tree"] = {p: rng.sample(TREE_CHILDREN, min(n_children, len(TREE_CHILDREN)))
                        for p in parents}

    def grow_tree() -> None:
        tree = spec.get("tree")
        if not tree:
            return
        for p in list(tree):
            extra = [c for c in TREE_CHILDREN if c not in tree[p]]
            tree[p] = tree[p] + rng.sample(extra, min(2, len(extra)))

    def add_table(n_rows: int) -> None:
        if spec.get("table") or not mid:
            return
        spec["table"] = [[rng.choice(TABLE_CELLS) for _ in TABLE_COLS]
                         for _ in range(n_rows)]

    def grow_table() -> None:
        if not spec.get("table"):
            return
        spec["table"] = spec["table"] + [[rng.choice(TABLE_CELLS) for _ in TABLE_COLS]]

    def add_menu(n_menus: int) -> None:
        """合并菜单：模板的菜单项要留着（可能是 oracle 目标），只补不删。"""
        if not mid:
            return
        menu = dict(spec.get("menu") or {})
        for name, items in list(MENU_POOL.items())[:n_menus]:
            have = menu.get(name, [])
            menu[name] = have + [i for i in items if i not in have]
        spec["menu"] = menu

    def add_buttons(n: int) -> int:
        """补按钮，返回实际加了几个（池子空了就加不动）。"""
        if not mid:
            return 0
        have = {name for name, _ in spec.get("buttons", [])}
        extra = [b for b in DISTRACTOR_BUTTONS if b not in have]
        rng.shuffle(extra)
        picked = extra[:max(0, n)]
        for name in picked:
            spec.setdefault("buttons", []).append((name, True))
        return len(picked)

    def add_icon_buttons(n: int) -> None:
        """无名图标控件：对齐真实桌面 11% 的无名元素比例。

        真实那批是**窗口按钮**（最小化/最大化/关闭，实测 bbox 69×60 在屏幕右上角）：
        UIA Name 为空，但像素里有图形。图形由 `mock_env._glyph_icon` 画
        （2026-09-19 owner 报「我没看到纯图标」后才补的——之前只造了空白按钮）。
        """
        if not mid:
            return
        for _ in range(max(0, n)):
            spec.setdefault("buttons", []).append(("", True))

    def add_fields(n: int) -> None:
        """输入框最多 3 个：它们从 field_y0 往下排，多到压住按钮区就会叠控件。"""
        if not mid:
            return
        fields = list(spec.get("fields", []))
        room = max(0, 3 - len(fields))
        pool = [f for f in DISTRACTOR_FIELDS if f not in fields]
        rng.shuffle(pool)
        spec["fields"] = fields + pool[:max(0, min(n, room))]

    # 无名图标控件先加（它是对齐真实分布的形状，不是填充料——不能被"够了就停"跳过）
    add_icon_buttons(DISTRACTOR_ICONS_PER_EPISODE if big else 4)

    # 先铺"面板形状"（树/表/菜单/输入）——它们是真实应用的结构骨架
    for step in (lambda: add_tree(4, 4), lambda: add_table(4), lambda: add_menu(2),
                 lambda: add_fields(1), lambda: grow_tree(), lambda: grow_table(),
                 lambda: add_menu(4), lambda: add_fields(2)):
        if est() >= size:
            break
        step()

    # 再用按钮补到目标：真实桌面 76% 的可见元素就是按钮（traces_full 实测 67/步），
    # 而按钮全在视口内、不受虚拟化影响，是唯一能可靠堆量的类型。
    # 2026-09-19 之前这里是"每档只走一遍"的阶梯，用虚高的估算器判停 →
    # 窗口实际只有 ~52 个可见控件，却以为已经有 85 了。
    for _ in range(40):
        if est() >= size:
            break
        if not add_buttons(4):
            break   # 池子空了：加不动了，如实停在当前量级
    if not mid:  # 小窗口不做任何加料（保留 v2 那样的干净分布做对照）
        spec.update({"size": (460, 340)})
    return spec


def data_names(spec: dict) -> set[str]:
    """数据类名字（列表项/树节点/表格单元/菜单项/下拉选项）。

    这些名字会出现在 goal 里（"删除列表中的「苹果」"），改名会让 goal 与控件对不上；
    而且同名控件可能同时出现在数据与按钮里，改一个会让 oracle 的执行目标歧义。
    所以双语改名只碰"非数据"控件。
    """
    names: set[str] = set()
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
    if spec.get("combo"):
        names |= set(spec["combo"]["items"])
    return names


def bilingual(spec: dict, steps: list, rng: random.Random, prob: float = 0.5) -> tuple[dict, list]:
    """按概率把一部分控件改成英文名，oracle 步骤同步改（标签仍是 gold）。

    这一步是冲着"中文目标 → 英文控件"这个真实语义鸿沟去的（计算器的 Multiply by）。
    改名映射是**确定性**的（保存→Save 每次都一样），所以模型能学到这个映射，
    而不是被随机噪声干扰。
    """
    if rng.random() >= prob:
        return spec, steps
    spec = dict(spec)
    protected = data_names(spec)
    # 现有全部控件名：改名不能造出重名控件，否则 oracle 按名字找目标会找错那个
    # （v3 的干扰池里本来就有 "Save"/"Edit" 这类英文名，撞名是真会发生的）
    taken = {n for n, _ in spec.get("buttons", [])} | set(spec.get("fields", [])) \
        | set(spec.get("checks", [])) | protected
    if spec.get("radios"):
        taken |= set(spec["radios"]["items"])
    if spec.get("combo"):
        taken.add(spec["combo"]["name"])
    rename: dict[str, str] = {}

    def pick(name: str) -> str:
        if not name or name in protected or name in rename:
            return rename.get(name, name)
        new = EN_RENAME.get(name)
        if new is None and rng.random() < 0.35:
            new = rng.choice(["Apply", "Proceed", "Confirm", "Dismiss", "Run",
                              "Start", "Stop", "Retry", "Open", "New"])
        if new and new != name and new not in protected and new not in taken:
            rename[name] = new
            taken.add(new)
        return rename.get(name, name)

    spec["buttons"] = [(pick(n), en) for n, en in spec.get("buttons", [])]
    spec["fields"] = [pick(f) for f in spec.get("fields", [])]
    spec["checks"] = [pick(c) for c in spec.get("checks", [])]
    if spec.get("readonly_fields"):
        spec["readonly_fields"] = [rename.get(f, f) for f in spec["readonly_fields"]]
    if spec.get("radios"):
        spec["radios"] = {"items": [pick(i) for i in spec["radios"]["items"]]}
    if spec.get("combo"):
        spec["combo"] = {"name": pick(spec["combo"]["name"]),
                         "items": list(spec["combo"]["items"])}
    if spec.get("select_enables"):
        spec["select_enables"] = [rename.get(n, n) for n in spec["select_enables"]]
    if spec.get("required_enables"):
        req, btn = spec["required_enables"]
        spec["required_enables"] = (rename.get(req, req), rename.get(btn, btn))
    if spec.get("dialog_on_click"):
        trigger, btns = spec["dialog_on_click"]
        spec["dialog_on_click"] = (rename.get(trigger, trigger),
                                   [rename.get(b, b) for b in btns])
    steps = [(a, rename.get(t, t) if t else t, v) for a, t, v in steps]
    return spec, steps


def semantic_goal(goal: str, steps: list, rng: random.Random, prob: float = 0.7) -> str:
    """把"照抄控件名"的目标改成同义/上下文表述（数据类目标准确引用不动）。

    按步骤顺序找**第一个在 goal 里被引号引用的、且有语义说法的目标**
    （多步流程里第一个可点目标常常是输入框，真正要改写的是后面那个语义按钮，
    如"搜索框→搜索"里的"搜索"）。

    只替换**引号内的**出现：控件名常常同时是普通动词（goal "删除列表中的「苹果」"
    里的"删除"是动作，不是按钮），无脑 substring 替换会把句子改坏。
    """
    if rng.random() >= prob:
        return goal
    for _, target, _ in steps:
        if not target or target not in SEMANTIC_PHRASES:
            continue
        for quoted in (f"「{target}」", f"'{target}'", f'"{target}"'):
            if quoted in goal:
                return goal.replace(quoted, rng.choice(SEMANTIC_PHRASES[target]), 1)
    return goal


def synthesize_elements(spec: dict) -> list[dict]:
    """估算「UIA 会看到的」元素表（**离线尺度验收用**，不执行任何动作）。

    ⚠️ 2026-09-19 重写。旧版是**纯手算 spec**：把折叠的树节点、未展开的子菜单项、
    视口外的表格行统统算进去，实测虚报 **+31/回合**（TreeItem +16.2、MenuItem +12.7、
    DataItem +4.0），而 `densify` 又拿它当"够了，别再加控件"的判据 ——
    结果窗口里只有 ~56 个可见控件，却对外宣称"元素中位 87 对齐真实 84"（假的对齐）。
    更糟的是真实观测被 owner 桌面上的别的窗口垫高到 86，把假象坐实了。

    现在改成**建一个不显示的窗口，按 Qt 自己的可见性数一遍**：
    - 普通控件用 `isVisibleTo(win)` —— 顺手就处理了"非当前 Tab 页不算"这类规则；
    - 树节点：祖先链全展开才算（UIA 只暴露可见节点）；
    - 表格/列表：只算视口内的行（Qt 虚拟化，视口外的 UIA 也看不到）；
    - 菜单栏：只算顶层项（子菜单要展开才进 UIA）；下拉框同理只算它自己。
    与窗口构建同源，`densify` / 验收 / 实测三者不会再漂移。
    """
    win = _build_quiet_window(spec)
    try:
        return _uia_visible_elements(win)
    finally:
        from PySide6.QtWidgets import QApplication
        win.close()
        win.deleteLater()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()


def _build_quiet_window(spec: dict):
    """建一个**不显示**的 MockWindow（估算用；rng 固定，保证估算可复现）。"""
    import mock_env
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:
        QApplication([])
    return mock_env.MockWindow(random.Random(0), "estimate", spec)


def _uia_visible_elements(win) -> list[dict]:
    """一个已建好（未显示）的窗口 → UIA 会暴露的元素列表（名字/类型/占位 bbox）。"""
    from PySide6.QtWidgets import (QCheckBox, QComboBox, QLineEdit, QListWidget,
                                   QMenuBar, QPushButton, QRadioButton, QTableWidget,
                                   QTreeWidget)

    out: list[dict] = []

    def el(name, ctype, widget=None):
        bbox = [0, 0, 120, 30]
        if widget is not None:
            g = widget.geometry()
            bbox = [g.left(), g.top(), g.right(), g.bottom()]
        out.append({"id": len(out) + 1, "name": name, "control_type": ctype,
                    "bbox": bbox, "center": [(bbox[0] + bbox[2]) // 2,
                                             (bbox[1] + bbox[3]) // 2],
                    "window_title": "MockUI"})

    for cls, ctype in ((QPushButton, "ButtonControl"), (QLineEdit, "EditControl"),
                       (QCheckBox, "CheckBoxControl"), (QRadioButton, "RadioButtonControl")):
        for w in win.findChildren(cls):
            if w.isVisibleTo(win):
                name = w.accessibleName() or (w.text() if hasattr(w, "text") else "")
                el(name, ctype, w)

    for combo in win.findChildren(QComboBox):
        if combo.isVisibleTo(win):
            # 下拉项只在弹出层打开时才进 UIA → 只算它自己
            el(combo.accessibleName() or "", "ComboBoxControl", combo)

    for tree in win.findChildren(QTreeWidget):
        if not tree.isVisibleTo(win):
            continue
        tree.doItemsLayout()   # 见下方表格处的说明：不布局就算不出视口
        viewport = tree.viewport().rect()

        def walk(item):
            # 视口外的节点 Qt 不画、UIA 也不暴露；折叠子树的节点同理
            if tree.visualItemRect(item).intersects(viewport):
                el(item.text(0), "TreeItemControl")
            if item.isExpanded():
                for i in range(item.childCount()):
                    walk(item.child(i))

        for i in range(tree.topLevelItemCount()):
            walk(tree.topLevelItem(i))

    for table in win.findChildren(QTableWidget):
        if not table.isVisibleTo(win):
            continue
        # 这个窗口**没显示过**，item view 还没做布局 → viewport() 是初始小尺寸
        # （实测 82×12），按它判"视口内"几乎全被误杀（DataItem 少算 19）。
        # 先 doItemsLayout() 逼它算出真实视口（482×158），命中的单元数就与实测一致了。
        table.doItemsLayout()
        viewport = table.viewport().rect()
        for r in range(table.rowCount()):
            for c in range(table.columnCount()):
                item = table.item(r, c)
                if item is None:
                    continue
                # 视口外的行 Qt 不画，UIA 也看不到（虚拟化）
                if table.visualItemRect(item).intersects(viewport):
                    el(item.text(), "DataItemControl")

    for lst in win.findChildren(QListWidget):
        if not lst.isVisibleTo(win):
            continue
        lst.doItemsLayout()
        viewport = lst.viewport().rect()
        for r in range(lst.count()):
            item = lst.item(r)
            if lst.visualItemRect(item).intersects(viewport):
                el(item.text(), "ListItemControl")

    for bar in win.findChildren(QMenuBar):
        if bar.isVisibleTo(win):
            for act in bar.actions():   # 只算顶层（子菜单要展开才进 UIA）
                el(act.text(), "MenuItemControl")

    return out


def build_episode(rng: random.Random, index: int, target_elements: int = 85,
                  bilingual_prob: float = 0.5, semantic_prob: float = 0.8) -> tuple[dict, str, list]:
    """一个 v3 回合：v2 模板 → densify → bilingual → semantic goal。"""
    template = mock_env._WEIGHTED[(index - 1) % len(mock_env._WEIGHTED)]
    spec, goal, steps = template(rng)
    steps = list(steps)
    spec = densify(spec, rng, target_elements)
    spec, steps = bilingual(spec, steps, rng, bilingual_prob)
    goal = semantic_goal(goal, steps, rng, semantic_prob)
    return spec, goal, steps


def estimate_elements(spec: dict) -> int:
    """预计元素数 —— **就是 `synthesize_elements` 的长度**。

    2026-09-19 修：这里原先是另一个"纯数 spec 条目"的快速计数器，与
    `synthesize_elements` **并存**。densify 用前者判停、验收用后者报数，
    两者差了 30 多个元素（前者把折叠树节点/子菜单项/视口外单元格都算上），
    于是"densify 觉得够了"和"实际看得见几个"彻底脱钩 —— 这就是尺度对齐造假的根。
    现在只留一条真源：数得慢一点没关系，数错才是灾难。
    """
    return len(synthesize_elements(spec))


def scale_report(specs: list[tuple[dict, str]], ckpt: str = "") -> dict:
    """v3 观测尺度验收：元素数 / token 数 vs 真实桌面与 v2 的参照值。

    参照（2026-09-19 实测）：真实桌面 元素中位 84 / token 中位 2380；
    mock v2 训练集 元素中位 14 / token 中位 457。v3 的目标是贴近前者。
    """
    import statistics

    from training.common import serialize_observation

    tok = None
    if ckpt:
        try:
            from training.dataset import build_tokenizer

            tok = build_tokenizer(ckpt)
        except Exception:  # noqa: BLE001 - 估不了 token 就只报元素数
            tok = None

    n_elems, n_tokens = [], []
    for spec, goal in specs:
        els = synthesize_elements(spec)
        n_elems.append(len(els))
        if tok is not None:
            text = serialize_observation(goal, "MockUI", els, [])
            n_tokens.append(len(tok(text, truncation=True, max_length=4096)["input_ids"]))
    return {
        "n": len(specs),
        "elements_p50": statistics.median(n_elems) if n_elems else 0,
        "elements_min": min(n_elems) if n_elems else 0,
        "elements_max": max(n_elems) if n_elems else 0,
        "tokens_p50": statistics.median(n_tokens) if n_tokens else 0,
        "tokens_max": max(n_tokens) if n_tokens else 0,
        "ref_real": {"elements_p50": 84, "tokens_p50": 2380},
        "ref_v2": {"elements_p50": 14, "tokens_p50": 457},
    }


def _wait_for_user_to_leave(idle_s: float, ep: int) -> None:
    """owner 在用电脑就等（"可以跑，但别影响我正常使用"）——开窗前必过这道闸门。"""
    if not idle_s:
        return
    from screen_use.winidle import user_idle_seconds, wait_for_idle

    if user_idle_seconds() >= idle_s:
        return
    print(f"[idle-gate] 检测到用户在用电脑，量产暂停（第 {ep} 回合前）……", flush=True)
    t0 = time.time()
    waited = wait_for_idle(idle_s, poll_s=5.0)
    print(f"[idle-gate] 用户空闲 {waited + idle_s:.0f}s，继续量产"
          f"（本回合等了 {time.time() - t0:.0f}s）", flush=True)


def _abort_on_empty_observation(ep: int, episode_id: str, n_elem: int,
                                out_dir: Path) -> None:
    """观测到 0 个控件 → 窗口对 UIA 根本不可见，立刻熔断。

    什么时候会这样：锁屏后 UIA 可能枚举不到窗口、平台插件不是真窗口（offscreen）、
    桌面被别的应用独占。此时继续跑只会量产一堆「元素表为空」的垃圾——
    而空表喂进训练 = 教模型在什么都没有的屏幕上下判断，比不跑还坏。
    所以删掉这一回合的文件再抛错，让流水线响亮地停下（断点续跑不会捡回垃圾）。
    """
    if n_elem:
        return
    bad = out_dir / f"{episode_id}.jsonl"
    bad.unlink(missing_ok=True)
    raise SystemExit(
        f"第 {ep} 回合观测到 0 个控件：窗口对 UIA 不可见"
        f"（锁屏？平台插件？桌面被独占？）。已删除 {bad.name} 并中止——"
        f"继续跑只会量产空观测的垃圾数据。")


def _icon_controls_lost_their_blank_name(spec: dict, observed: list[dict]) -> bool:
    """spec 掺了无名按钮，观测里却一个无名元素都没有 → 图标把 UIA Name 撑起来了？

    纯图标控件的全部意义就是「Name 为空」（真实域实测 10.2%，全是 ButtonControl）。
    如果给按钮 `setIcon` 之后 UIA 开始报名字，这条分布对齐就**静默失效**了
    （mock 从 10.4% 掉到 0%，模型又回到"每个控件都有名字"的世界）。
    只在 spec 真有无名按钮的回合检查，避免误报。
    """
    want = sum(1 for n, _ in spec.get("buttons", []) if not str(n).strip())
    if not want:
        return False
    return not any(not str(e.get("name", "")).strip() for e in observed)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start", type=int, default=1, help="断点续跑（rng 序列一致）")
    parser.add_argument("--target-elements", type=int, default=85,
                        help="目标元素数（真实桌面中位 84；v2 只有 14）")
    parser.add_argument("--bilingual-prob", type=float, default=0.5)
    parser.add_argument("--out", default="", help="默认 ~/.screen_use/traces_mock_v3")
    parser.add_argument("--estimate-only", action="store_true",
                        help="不开窗：只报观测尺度（元素/token）并与真实/v2 对照")
    parser.add_argument("--idle-gate-s", type=float, default=45.0,
                        help="用户空闲满这么多秒才继续量产（人在用就暂停；0=不设闸门）")
    parser.add_argument("--ckpt", default="checkpoints/jev-Qwen3-0.6B-e3",
                        help="算 token 用的 tokenizer 来源（--estimate-only）")
    args = parser.parse_args()

    if args.estimate_only:
        rng = random.Random(args.seed)
        specs = []
        for ep in range(1, 41):
            spec, goal, _ = build_episode(rng, ep, args.target_elements,
                                          args.bilingual_prob)
            specs.append((spec, goal))
        import json

        print(json.dumps(scale_report(specs, args.ckpt), ensure_ascii=False, indent=1))
        return

    from PySide6.QtWidgets import QApplication

    rng = random.Random(args.seed)
    app = QApplication(sys.argv)
    out_dir = Path(args.out).expanduser() if args.out else _default_home() / "traces_mock_v3"
    runner = mock_env.EpisodeRunner(out_dir)
    total = {"episodes": 0, "steps": 0, "ok": 0, "elements": []}
    icon_warned = False

    for ep in range(1, args.episodes + 1):
        spec, goal, steps = build_episode(rng, ep, args.target_elements,
                                          args.bilingual_prob)
        if ep < args.start:
            continue  # 保持 rng 消费一致
        # 进闸门前清场：gate 里 sleep 会阻塞 Qt 事件循环，此时屏幕上**绝不允许**
        # 有 mock 窗口（否则 Windows 会把它标成「(未响应)」挂在 owner 桌面上）
        swept = mock_env.sweep_stale_windows()
        if swept:
            print(f"[janitor] 进闸门前清掉 {swept} 个残留 mock 窗口", flush=True)
        _wait_for_user_to_leave(args.idle_gate_s, ep)
        episode_id = f"mock-v3-{args.seed}-{ep:04d}"
        win = mock_env.MockWindow(rng, episode_id, spec)
        try:
            win.show()
            app.processEvents()
            time.sleep(0.4)
            stats = runner.run(win, goal, steps, episode_id)
            observed = runner._observe(episode_id)
        finally:
            # 确认框/弹层随窗口一起收（2026-09-19 卡死事故，见 MockWindow.teardown）
            win.teardown()
            app.processEvents()
        n_elem = len(observed)
        _abort_on_empty_observation(ep, episode_id, n_elem, out_dir)
        if not icon_warned and _icon_controls_lost_their_blank_name(spec, observed):
            print(f"[warn] 第 {ep} 回合的 spec 里有无名图标按钮，但观测表里一个无名元素都没有"
                  f"——图标可能被 UIA 当成了 Name，11% 无名这条分布对齐已失效，"
                  f"查 _glyph_icon / accessibleName", flush=True)
            icon_warned = True
        total["elements"].append(n_elem)
        total["episodes"] += 1
        total["steps"] += stats["steps"]
        total["ok"] += stats["ok"]
        if ep % 25 == 0:
            avg = sum(total["elements"]) / len(total["elements"])
            # 自检：实测均值必须落在目标附近。2026-09-19 的教训是"报出来的尺度"
            # 可以因为估算器虚高而完全是假的（报 87、实际 56），所以这里只认**实测**。
            off = abs(avg - args.target_elements)
            flag = ""
            if off > 0.2 * args.target_elements:
                flag = (f"  ⚠️ 偏离目标 {args.target_elements} 达 {off:.0f} 个"
                        f"（估算器/字体/布局可能又漂了，别信离线数字）")
            print(f"[{ep}/{args.episodes}] steps={total['steps']} ok={total['ok']} "
                  f"元素均值={avg:.0f}{flag}", flush=True)

    avg = sum(total["elements"]) / max(len(total["elements"]), 1)
    print(f"完成: {total['episodes']} episodes, {total['steps']} 步 "
          f"(oracle 成功 {total['ok']}), 元素均值 {avg:.0f} → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
