"""Debug：拆解"跨应用转移"环节，逐步验证。

环境：计算器预置 56×7=392，记事本空白打开。
阶段 1: 元素枚举是否同时覆盖两个窗口
阶段 2: 点击记事本元素是否能激活窗口（遮挡修复验证）
阶段 3: VLM 对转移任务的原始决策输出（完整打印）
阶段 4: run_task 完整跑，校验记事本内容
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uiautomation as auto

from screen_use import ScreenUse

auto.SetGlobalSearchTimeout(5)

SCRATCH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", f"scratch_debug_{int(time.time())}.txt"
))


def find_notepad():
    for w in auto.GetRootControl().GetChildren():
        try:
            if w.ClassName == "Notepad" and "scratch_debug" in (w.Name or ""):
                return w
        except Exception:
            pass
    return None


class RawLoggingProvider:
    """包装真实 provider，打印原始响应。"""

    def __init__(self, inner):
        self._inner = inner

    def pick_element(self, *a, **kw):
        return self._inner.pick_element(*a, **kw)

    def ask_about_screen(self, image_base64, question):
        raw = self._inner.ask_about_screen(image_base64, question)
        print(f"\n    [RAW VLM] {raw[:300]!r}\n", flush=True)
        return raw


def main() -> None:
    tools = ScreenUse()

    # ---- 环境准备 ----
    while True:
        leftover = find_notepad()
        if not leftover:
            break
        leftover.SetActive()
        time.sleep(0.3)
        leftover.GetWindowPattern().Close()
        time.sleep(0.8)
        r = tools.click_element("Don't save")
        if not r.get("found"):
            tools.click_element("不保存")
        time.sleep(0.5)

    open(SCRATCH, "w").write("")
    os.startfile(SCRATCH)
    time.sleep(3)
    np_win = find_notepad()
    np_win.GetTransformPattern().Move(20, 20)
    time.sleep(0.3)

    os.startfile("calc.exe")
    time.sleep(2.5)
    calc = auto.WindowControl(searchDepth=1, SubName="Calculator")
    calc.SetActive()
    time.sleep(0.3)
    calc.GetTransformPattern().Move(1700, 20)
    time.sleep(0.5)
    for step in ["Clear", "Five", "Six", "Multiply by", "Seven", "Equals"]:
        tools.click_element(step)
        time.sleep(0.2)
    print("环境就绪：计算器=392，记事本空\n", flush=True)

    # ---- 阶段 1：元素枚举 ----
    print("=== 阶段1: 元素枚举 ===", flush=True)
    elements = tools.list_ui_elements()
    by_window = {}
    for el in elements:
        by_window.setdefault(el["window_title"], []).append(el)
    for title, els in by_window.items():
        print(f"  [{title[:30]}] {len(els)} 个元素", flush=True)
    np_elements = [e for t, els in by_window.items() if "scratch_debug" in t for e in els]
    calc_elements = [e for t, els in by_window.items() if "Calculator" in t for e in els]
    print(f"  记事本元素: {len(np_elements)}, 计算器元素: {len(calc_elements)}", flush=True)

    # ---- 阶段 2：点击记事本元素激活窗口 ----
    print("\n=== 阶段2: 点击记事本元素 → 窗口激活验证 ===", flush=True)
    if np_elements:
        target = next((e for e in np_elements if e["control_type"] == "DocumentControl"), np_elements[0])
        print(f"  目标: #{target['id']} [{target['control_type']}] {target['name']!r}", flush=True)
        tools.click_element_id(target["id"])
        time.sleep(1)
        fg = auto.GetForegroundControl()
        fg_name = fg.Name if fg else "?"
        print(f"  前台窗口现在是: {fg_name!r}", flush=True)
        print(f"  激活{'成功' if 'scratch_debug' in (fg_name or '') else '失败'}", flush=True)

    # ---- 阶段 3：VLM 原始决策 ----
    print("\n=== 阶段3: VLM 原始决策输出 ===", flush=True)
    tools2 = ScreenUse(provider=RawLoggingProvider(tools.provider))
    shot = tools2.screenshot(annotate=True)
    from screen_use.agent import _LOOP_PROMPT
    legend = "\n".join(
        f"{el['id']}: [{el['window_title']}] [{el['control_type']}] {el['name'] or '(无名)'}"
        for el in shot["elements"]
    )
    prompt = _LOOP_PROMPT.format(
        goal="把计算器显示屏上的结果数字输入到记事本窗口里",
        legend=legend, history="(无)", foreground="scratch_debug",
        experience="", reflection="",
        width=shot["width"], height=shot["height"], scale=shot["scale"],
    )
    raw = tools2.provider.ask_about_screen(shot["image_base64"], prompt)

    # ---- 阶段 4：完整 run_task ----
    print("\n=== 阶段4: run_task 完整执行 ===", flush=True)

    def log(step):
        print(f"[step {step.index}] {step.thought[:50]!r} -> {step.action} {step.args} ok={step.ok}", flush=True)

    result = tools.run_task(
        "把计算器显示屏上的结果数字输入到记事本窗口里",
        max_steps=6, on_step=log,
    )
    print(f"done={result['done']} result={result['result']!r}", flush=True)

    # 校验
    win = find_notepad()
    if win:
        doc = win.DocumentControl(searchDepth=10)
        content = doc.GetTextPattern().DocumentRange.GetText(-1)
        print(f"记事本最终内容: {content!r}", flush=True)
        print("转移验证:", "✅ 通过" if "392" in content else "❌ 失败", flush=True)
        win.GetWindowPattern().Close()
        time.sleep(1)
        r = tools.click_element("Don't save")
        if not r.get("found"):
            tools.click_element("不保存")
    if os.path.exists(SCRATCH):
        os.remove(SCRATCH)


if __name__ == "__main__":
    main()
