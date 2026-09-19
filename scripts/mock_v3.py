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

# 干扰控件：真实应用里大量存在的、跟当前任务无关的东西
DISTRACTOR_BUTTONS = [
    "File", "Edit", "View", "Help", "Refresh", "Export CSV", "Import", "Settings",
    "Profile", "Notifications", "Sync", "Filters", "Sort", "Columns", "Zoom In",
    "Zoom Out", "Fullscreen", "Share", "Print Preview", "Duplicate", "Archive",
    "Mark as Read", "Pin", "Collapse All", "Expand All",
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
    # 屏幕 1280×800，窗口 860×640 @ y=20 正好装下：面板 40–270 / 输入 280–420 /
    # 勾选单选 420–580 / 按钮 590–635。子节点必须展开（折叠的 UIA 看不到）。
    if big or mid:
        spec.update({
            "size": (860, 700), "menu_pos": (0, 0, 860, 28),
            "tree_pos": (700, 40, 150, 190), "list_pos": (520, 40, 170, 190),
            "table_pos": (10, 40, 500, 190), "field_y0": 250,
            "combo_pos": (700, 255, 140, 30),
            "button_pos": (10, 0), "button_cols": 8,
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

    def add_buttons(n: int) -> None:
        if not mid:
            return
        have = {name for name, _ in spec.get("buttons", [])}
        extra = [b for b in DISTRACTOR_BUTTONS if b not in have]
        rng.shuffle(extra)
        for name in extra[:max(0, n)]:
            spec.setdefault("buttons", []).append((name, True))

    def add_icon_buttons(n: int) -> None:
        """无名图标按钮：对齐真实桌面 11% 的无名元素比例（Qt 空文本按钮 UIA Name 为空）。"""
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

    # 按性价比排序逐档加料，够了就停（避免把 84 元素的真实分布做成 120）
    for step in (lambda: add_tree(4, 4), lambda: add_table(4), lambda: add_menu(2),
                 lambda: add_buttons(5), lambda: add_fields(1),
                 lambda: grow_tree(), lambda: add_menu(4), lambda: add_buttons(6),
                 lambda: grow_table(), lambda: add_fields(2)):
        if est() >= size:
            break
        step()
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
    """按 spec 合成「UIA 会看到的」元素表（**离线估算尺度用**，不执行任何动作）。

    面板内部的行/单元格（树节点、表格单元、列表项、菜单项）在 Qt 里是 model item
    而不是 QWidget，真实运行时由 UIA 逐个暴露；这里按同样的名字/类型合成出来，
    bbox 用占位数——只为把「元素数 / token 数」量到真实桌面的量级上做验收，
    不参与任何轨迹落盘。
    """
    def el(i, name, ctype, x, y):
        return {"id": i, "name": name, "control_type": ctype,
                "bbox": [x, y, x + 120, y + 30], "center": [x + 60, y + 15],
                "window_title": "MockUI"}

    out: list[dict] = []
    n = 0
    y = 260
    for name in spec.get("fields", []):
        n += 1
        out.append(el(n, name, "EditControl", 20, y))
        y += 45
    for name in spec.get("checks", []):
        n += 1
        out.append(el(n, name, "CheckBoxControl", 20, y))
        y += 32
    for name in (spec.get("radios") or {}).get("items", []):
        n += 1
        out.append(el(n, name, "RadioButtonControl", 250, y))
    if spec.get("combo"):
        n += 1
        out.append(el(n, spec["combo"]["name"], "ComboBoxControl", 500, 260))
        for item in spec["combo"]["items"]:
            n += 1
            out.append(el(n, item, "ListItemControl", 500, y))
    if spec.get("list"):
        n += 1
        out.append(el(n, "项目列表", "ListControl", 240, 40))
        for item in spec["list"]["items"]:
            n += 1
            out.append(el(n, item, "ListItemControl", 240, y))
    for parent, children in (spec.get("tree") or {}).items():
        n += 1
        out.append(el(n, parent, "TreeItemControl", 10, 40))
        for child in children:
            n += 1
            out.append(el(n, child, "TreeItemControl", 30, y))
    for row in spec.get("table") or []:
        for cell in row:
            n += 1
            out.append(el(n, cell, "DataItemControl", 470, y))
    for menu, items in (spec.get("menu") or {}).items():
        n += 1
        out.append(el(n, menu, "MenuItemControl", 0, 0))
        for item in items:
            n += 1
            out.append(el(n, item, "MenuItemControl", 0, 20))
    for name, _enabled in spec.get("buttons", []):
        n += 1
        out.append(el(n, name, "ButtonControl", 20, y))
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
    """预计元素数（与 MockWindow 实际暴露的可交互控件大致对应）。"""
    n = len(spec.get("fields", [])) + len(spec.get("buttons", []))
    n += len(spec.get("checks", []))
    n += len(spec.get("radios", {}).get("items", [])) if spec.get("radios") else 0
    n += 1 if spec.get("combo") else 0
    n += len(spec.get("list", {}).get("items", [])) if spec.get("list") else 0
    n += sum(1 + len(c) for c in (spec.get("tree") or {}).values())
    table = spec.get("table") or []
    n += (1 if table else 0) + sum(len(r) for r in table)        # 表格 + 单元格
    n += sum(1 + len(v) for v in (spec.get("menu") or {}).values())  # 菜单名 + 菜单项
    return n


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

    for ep in range(1, args.episodes + 1):
        spec, goal, steps = build_episode(rng, ep, args.target_elements,
                                          args.bilingual_prob)
        if ep < args.start:
            continue  # 保持 rng 消费一致
        _wait_for_user_to_leave(args.idle_gate_s, ep)
        episode_id = f"mock-v3-{args.seed}-{ep:04d}"
        win = mock_env.MockWindow(rng, episode_id, spec)
        win.show()
        app.processEvents()
        time.sleep(0.4)
        stats = runner.run(win, goal, steps, episode_id)
        total["elements"].append(len(runner._observe(episode_id)))
        win.close()
        app.processEvents()
        total["episodes"] += 1
        total["steps"] += stats["steps"]
        total["ok"] += stats["ok"]
        if ep % 25 == 0:
            avg = sum(total["elements"]) / len(total["elements"])
            print(f"[{ep}/{args.episodes}] steps={total['steps']} ok={total['ok']} "
                  f"元素均值={avg:.0f}", flush=True)

    avg = sum(total["elements"]) / max(len(total["elements"]), 1)
    print(f"完成: {total['episodes']} episodes, {total['steps']} 步 "
          f"(oracle 成功 {total['ok']}), 元素均值 {avg:.0f} → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
