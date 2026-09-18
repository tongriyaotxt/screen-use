"""desktop-jev 阶段 1：真实任务轨迹采集。

设计 6 个常用任务（计算器/记事本，安全可逆），用 kimi-code VLM 跑 run_task，
逐步落盘结构化轨迹（TraceRecorder → ~/.screen_use/traces_full/）。

用法（项目根，.venv）：
    ./.venv/Scripts/python.exe scripts/collect_traces.py [任务序号...]

被操作的应用窗口自动贴到屏幕右半边（Win+Right），方便左侧实时观察 agent 进程。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

os.environ.setdefault("VISION_PROVIDER", "kimi-code")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pyautogui

from screen_use.config import Settings
from screen_use.tools import ScreenUse

# ---- 任务定义 ----
# apps: 需预先启动的应用；goal 不含"打开应用"（脚本已预启动并贴边）
TASKS = [
    {
        "name": "calc-basic",
        "apps": ["calc"],
        "goal": "用计算器计算 78 乘以 9，算出结果",
    },
    {
        "name": "calc-chain",
        "apps": ["calc"],
        "goal": "用计算器计算 (256 加 144) 除以 8，算出结果",
    },
    {
        "name": "notepad-type",
        "apps": ["notepad"],
        "goal": "在记事本中输入这句话：今天天气不错，适合写代码。",
    },
    {
        "name": "notepad-multiline",
        "apps": ["notepad"],
        "goal": "在记事本中输入三行内容：苹果、香蕉、橘子，每个词单独一行",
    },
    {
        "name": "cross-calc-to-notepad",
        "apps": ["calc", "notepad"],
        "goal": "用计算器计算 125 乘以 8，然后把计算结果输入到记事本中",
    },
    {
        "name": "cross-calc-to-notepad-2",
        "apps": ["calc", "notepad"],
        "goal": "用计算器计算 1024 除以 16，把结果写到记事本里，再在最后加上“是答案”三个字",
    },
]

_APPS = {
    "calc": ("calc.exe", "CalculatorApp.exe"),
    "notepad": ("notepad.exe", "Notepad.exe"),
}

MAX_STEPS = 25


def cleanup() -> None:
    """关掉采集用的应用，保证每个任务初始状态干净；鼠标移回屏幕中央复位 fail-safe。"""
    for _, proc in _APPS.values():
        subprocess.run(["taskkill", "/f", "/im", proc],
                       capture_output=True, check=False)
    pyautogui.moveTo(1280, 800)  # 屏幕中央：避免鼠标卡角落导致后续点击全部 fail-safe
    time.sleep(1)


def launch_and_snap_right(app_key: str) -> None:
    """启动应用并贴到屏幕右半边（Win+Right），让出左边给观察者。"""
    launch_cmd, _ = _APPS[app_key]
    subprocess.Popen(["cmd", "/c", "start", "", launch_cmd],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)  # 等窗口起来并成为前台
    pyautogui.hotkey("win", "right")
    time.sleep(1)


def run_task(task: dict, idx: int) -> bool:
    print(f"\n{'=' * 60}\n[{idx}] {task['name']}: {task['goal']}\n{'=' * 60}", flush=True)
    cleanup()
    for app in task["apps"]:
        launch_and_snap_right(app)

    tools = ScreenUse(settings=Settings(vision_provider="kimi-code"))

    def on_step(step):
        status = "OK" if step.ok else "FAIL"
        print(f"  step {step.index:02d} {status} {step.action} {step.args} {step.detail}",
              flush=True)

    t0 = time.time()
    result = tools.run_task(task["goal"], max_steps=MAX_STEPS, on_step=on_step)
    dt = time.time() - t0
    print(f"  → done={result['done']} 结果={result['result']!r} "
          f"步数={len(result['steps'])} 耗时={dt:.1f}s", flush=True)
    cleanup()
    return result["done"]


def main() -> None:
    indices = [int(a) for a in sys.argv[1:]] or list(range(len(TASKS)))
    print(f"VLM: kimi-code | 任务: {[TASKS[i]['name'] for i in indices]}", flush=True)
    ok = 0
    for n, i in enumerate(indices, 1):
        try:
            ok += run_task(TASKS[i], n)
        except Exception as e:  # 单个任务失败不中断采集
            print(f"  !! 任务异常: {e}", flush=True)
            cleanup()
    print(f"\n完成 {ok}/{len(indices)} 个任务。轨迹在 ~/.screen_use/traces_full/", flush=True)


if __name__ == "__main__":
    main()
