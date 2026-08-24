"""Debug：复现"算完数就卡住"——带完整决策日志的循环运行。

场景：计算器已算出 56×7=392，记事本已打开，
任务只剩后半段："把计算器里的结果输入到记事本"。
逐步打印 VLM 的原始决策，观察卡在哪。
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


def main() -> None:
    tools = ScreenUse()

    # 打开记事本放左边
    open(SCRATCH, "w").write("")
    os.startfile(SCRATCH)
    time.sleep(3)
    np_win = find_notepad()
    np_win.GetTransformPattern().Move(10, 10)
    time.sleep(0.3)

    # 打开计算器放右边（不算，让 agent 从头做）
    os.startfile("calc.exe")
    time.sleep(2.5)
    calc = auto.WindowControl(searchDepth=1, SubName="Calculator")
    calc.SetActive()
    time.sleep(0.5)
    print("=== 环境就绪，开始完整任务 debug ===", flush=True)

    def log(step):
        print(
            f"[step {step.index}] thought={step.thought!r}\n"
            f"          action={step.action} args={step.args} ok={step.ok} detail={step.detail!r}",
            flush=True,
        )

    result = tools.run_task(
        "在计算器中计算 56 乘以 7，然后把结果数字输入到记事本窗口里",
        max_steps=12,
        on_step=log,
    )
    print(f"=== done={result['done']} result={result['result']!r} ===", flush=True)

    # 清理
    win = find_notepad()
    if win:
        win.GetWindowPattern().Close()
        time.sleep(1)
        r = tools.click_element("Don't save")
        if not r.get("found"):
            tools.click_element("不保存")
    if os.path.exists(SCRATCH):
        os.remove(SCRATCH)


if __name__ == "__main__":
    main()
