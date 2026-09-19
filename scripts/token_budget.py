"""每步决策的 token 预算 / Pareto 网格（prefill 延迟就由它决定）。

用法（项目根）：
    ./.venv/Scripts/python.exe scripts/token_budget.py

口径：
- 文本侧用本地 Qwen 系 tokenizer 近似（同 BPE 家族；Qwen2.5-VL 实际值差几个点）
- 视觉侧按 Qwen2.5-VL 公式：patch 14 + merge 2 → 每视觉 token 覆盖 28×28 像素，
  smart_resize 到 28 的整数倍且不超 max_pixels
- **两种形态分开列**：路线 A（零样本，prompt 里要带动作说明书）vs
  路线 B（微调后，说明书长在权重里，一个字不写）
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from screen_use.observation import (format_history_entry, serialize_elements,  # noqa: E402
                                    serialize_observation)

HOME = pathlib.Path.home() / ".screen_use"
TOKENIZER = "checkpoints/jev-Qwen3-0.6B-e3"
SCREEN = (1280, 800)          # 物理分辨率（决定截图的宽高比）
GOAL = "打开计算器，算 78 乘以 9，把结果记到记事本里"


def load_sample() -> dict:
    for f in sorted((HOME / "traces_mock_v3").glob("*.jsonl")):
        steps = [json.loads(line) for line in
                 f.read_text(encoding="utf-8").splitlines() if line.strip()]
        if steps and len(steps[0].get("elements", [])) > 80:
            return steps[0]
    raise SystemExit("没找到样本：先跑一次 mock v3 量产")


def visual_tokens(max_side: int) -> int:
    """Qwen2.5-VL 的视觉 token 数（28×28 像素/token）。"""
    ratio = SCREEN[0] / SCREEN[1]
    w, h = max_side, int(max_side / ratio)
    return round(w / 28) * round(h / 28)


def action_manual_tokens(tok) -> int:
    """路线 A 才需要的动作说明书（旧 _LOOP_PROMPT 里那一段）。"""
    manual = """
可用动作（只输出一个 JSON，不要输出其他内容）：
- {"thought": "分析当前状态", "action": "click_id", "id": 元素编号}  点击某个编号元素
- {"thought": "...", "action": "click", "x": 横坐标, "y": 纵坐标}  点击坐标（元素表中没有时）
- {"thought": "...", "action": "type", "text": "文本"}}
- {"thought": "...", "action": "hotkey", "keys": "ctrl+s"}}
- {"thought": "...", "action": "press", "key": "enter"}}
- {"thought": "...", "action": "scroll", "clicks": -3}}
- {"thought": "...", "action": "wait", "seconds": 1}}
- {"thought": "...", "action": "done", "result": "任务完成的总结"}}
关键规则：0. 不要输出思考过程，直接只输出 JSON。1. 每步只做一个动作；
2. 输入文本前必须先点击目标输入区域获得焦点；3. done 前必须确认截图里能看到结果；
4. 切换窗口直接点击候选元素；5. 键盘连续操作必须用 batch 一次提交。
"""
    return len(tok(manual)["input_ids"])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    step = load_sample()
    els = step["elements"]
    tok = AutoTokenizer.from_pretrained(str(ROOT / TOKENIZER))

    def n(text: str) -> int:
        return len(tok(text)["input_ids"])

    # 状态位不在旧轨迹里 → 按真实分布注入（8% 禁用 / 5% 勾选 / 3% 展开）
    els_flagged = []
    for i, e in enumerate(els):
        e2 = dict(e)
        if i % 13 == 0:
            e2["disabled"] = True
        if i % 19 == 0:
            e2["checked"] = True
        if i % 31 == 0:
            e2["expanded"] = True
        els_flagged.append(e2)

    print(f"样本 {len(els)} 元素 · 屏幕 {SCREEN[0]}x{SCREEN[1]}\n")

    print("== 元素表（文本侧） ==")
    styles = {
        "① legend（现状，全类型名）": serialize_elements(els, "legend"),
        "③ bbox（旧 Jev，反例）": serialize_elements(els, "bbox"),
        "④ min（id+名字）": serialize_elements(els, "min"),
        "④b tight（类型+状态）": serialize_elements(els_flagged, "tight"),
    }
    for k, v in styles.items():
        print(f"  {k:28s} {n(v):6d} token   {n(v)/len(els):5.1f}/元素")

    print("\n== 历史（单行压缩；旧实现是中文长句） ==")
    hist_old = "\n".join(f"第{i}步: click_id({i}) 点击了某个按钮，执行成功"
                         for i in range(1, 9))
    print(f"  旧实现 8 步: {n(hist_old):5d} token   {n(hist_old)/8:.1f}/步")
    for k in (2, 4, 6, 8):
        entries = [format_history_entry(i, "CLICK",
                                        {"id": 40 + i}, text="", ok=(i % 5 != 0))
                   for i in range(1, k + 1)]
        h = "\n".join(entries)
        print(f"  新实现 k={k}: {n(h):5d} token   {n(h)/k:.1f}/步"
              f"   例：{entries[-1]}")

    print("\n== 视觉侧（Qwen2.5-VL 公式） ==")
    for side in (896, 672, 448, 336):
        print(f"  最长边 {side:3d} → 视觉 token {visual_tokens(side):5d}")

    manual = action_manual_tokens(tok)
    print(f"\n== 动作说明书（只有路线 A 需要）: {manual} token ==")

    print("\n== 每步总预算（元素表 + goal/历史 + 视觉 [+ 说明书]） ==")
    print(f"{'形态':34s} {'@896':>7s} {'@448':>7s} {'@336':>7s}")
    hist4 = "\n".join(format_history_entry(i, "CLICK", {"id": 40 + i}, ok=True)
                      for i in range(1, 5))
    for name, body in (("① legend + 旧历史", serialize_elements(els, "legend")),
                       ("④ min + 压缩历史", serialize_elements(els, "min")),
                       ("④b tight + 压缩历史", serialize_elements(els_flagged, "tight"))):
        for label, extra in (("路线A(带说明书)", manual), ("路线B(无说明书)", 0)):
            base = n(body) + n(hist4) + n(f"T {GOAL}") + extra
            row = "  ".join(f"{base + visual_tokens(s):7d}" for s in (896, 448, 336))
            print(f"{name + ' ' + label:34s} {row}")

    print("\n注：后续步命中前缀缓存时，goal/说明书不再计入 → 只剩 元素表 + 视觉 + 历史尾巴")


if __name__ == "__main__":
    main()
