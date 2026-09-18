"""desktop-jev 阶段 1：人工驾驶采集器（oracle 模式）。

由调用者（Kimi CLI agent）做决策，本脚本负责：
- observe：枚举 UIA 元素（观察）
- step：先枚举元素（决策前观测）→ 落盘一步轨迹 → 执行动作 → 回填 ok/detail
  保证 (观测_t, 动作_t) 对齐，标签质量 = gold（oracle）
- begin/launch/kill/snap：任务生命周期与窗口管理

用法（项目根，.venv）：
    python scripts/drive.py observe
    python scripts/drive.py begin "任务目标"
    python scripts/drive.py step <taskfile> <idx> click_id 75
    python scripts/drive.py step <taskfile> <idx> type "文本"
    python scripts/drive.py step <taskfile> <idx> hotkey ctrl+a
    python scripts/drive.py step <taskfile> <idx> press enter
    python scripts/drive.py step <taskfile> <idx> done "结果总结"
    python scripts/drive.py launch calc|notepad   # 启动并贴右半屏
    python scripts/drive.py snap                   # 当前窗口贴右半屏
    python scripts/drive.py kill                   # 关掉采集应用+鼠标回中
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pyautogui

from screen_use.memory import TraceRecorder
from screen_use.tools import ScreenUse

_APPS = {
    "calc": ("calc.exe", "CalculatorApp.exe"),
    "notepad": ("notepad.exe", "Notepad.exe"),
}

_tools: ScreenUse | None = None


def tools() -> ScreenUse:
    global _tools
    if _tools is None:
        _tools = ScreenUse()  # 原子动作不需要 VLM
    return _tools


def observe() -> None:
    data = tools().list_ui_elements()
    print(f"前台: {data['foreground_title']}")
    for el in data["elements"]:
        print(f"#{el['id']} [{el['control_type']}] {el['name']!r} "
              f"bbox={el['bbox']} win={el['window_title']!r}")


def begin(goal: str) -> None:
    rec = TraceRecorder()
    task_id = rec.start_task(goal)
    print(rec.dir / f"{task_id}.jsonl")


def _append_step(taskfile: str, goal: str, idx: int, foreground: str,
                 elements: list[dict], action: str, args: dict,
                 ok: bool, detail: str) -> None:
    entry = {
        "goal": goal, "step_idx": idx, "foreground_title": foreground,
        "elements": elements, "action": action, "args": args,
        "ok": ok, "detail": detail, "ts": time.time(),
    }
    with open(taskfile, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def step(taskfile: str, idx: int, action: str, action_args: list[str]) -> None:
    t = tools()
    # 1) 决策前观测（与动作对齐的 observation_t）
    data = t.list_ui_elements()
    elements = data["elements"]
    foreground = data["foreground_title"] or ""

    goal = ""
    if Path(taskfile).exists():
        with open(taskfile, encoding="utf-8") as f:
            first = f.readline()
        if first.strip():
            goal = json.loads(first).get("goal", "")

    # 2) 解析并执行动作
    args: dict = {}
    ok, detail = True, ""
    try:
        if action == "click_id":
            args = {"id": int(action_args[0])}
            r = t.click_element_id(args["id"])
            detail = f"点击元素 #{args['id']}" if r.get("ok") else r.get("detail", "")
            ok = bool(r.get("ok"))
        elif action == "type":
            args = {"text": action_args[0]}
            t.type_text(args["text"])
            detail = f"输入 {args['text']!r}"
        elif action == "hotkey":
            args = {"keys": action_args[0]}
            t.hotkey(*action_args[0].split("+"))
            detail = f"快捷键 {action_args[0]}"
        elif action == "press":
            args = {"key": action_args[0]}
            t.press(action_args[0])
            detail = f"按键 {action_args[0]}"
        elif action == "scroll":
            args = {"clicks": int(action_args[0])}
            t.scroll(args["clicks"])
            detail = f"滚动 {args['clicks']}"
        elif action == "wait":
            args = {"seconds": float(action_args[0]) if action_args else 1.0}
            time.sleep(min(args["seconds"], 10))
            detail = "等待"
        elif action == "done":
            args = {"result": action_args[0] if action_args else "完成"}
            detail = "任务完成"
        else:
            raise ValueError(f"未知动作: {action}")
    except Exception as e:
        ok, detail = False, f"动作失败: {e}"

    # 3) 落盘（观测 + 决策 + 结果）
    _append_step(taskfile, goal, idx, foreground, elements, action, args, ok, detail)
    print(f"{'OK' if ok else 'FAIL'} step {idx} {action} {args} {detail} | 前台: {foreground}")


def launch(app: str) -> None:
    cmd, _ = _APPS[app]
    subprocess.Popen(["cmd", "/c", "start", "", cmd],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    pyautogui.hotkey("win", "right")
    time.sleep(1)
    print(f"launched {app} + snapped right")


def snap() -> None:
    pyautogui.hotkey("win", "right")
    print("snapped right")


def kill() -> None:
    for _, proc in _APPS.values():
        subprocess.run(["taskkill", "/f", "/im", proc],
                       capture_output=True, check=False)
    pyautogui.moveTo(1280, 800)
    print("killed + mouse centered")


def focus(name_sub: str) -> None:
    """把标题含 name_sub 的窗口激活到前台（环境准备，不计入轨迹）。"""
    import ctypes

    import uiautomation as auto
    for win in auto.GetRootControl().GetChildren():
        if win.ControlTypeName != "WindowControl":
            continue
        if name_sub.lower() in (win.Name or "").lower():
            hwnd = win.NativeWindowHandle
            user32 = ctypes.windll.user32
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
            time.sleep(0.5)
            print(f"focused: {win.Name!r}")
            return
    raise SystemExit(f"窗口未找到: {name_sub}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmd = sys.argv[1]
    if cmd == "observe":
        observe()
    elif cmd == "begin":
        begin(sys.argv[2])
    elif cmd == "step":
        step(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5:])
    elif cmd == "launch":
        launch(sys.argv[2])
    elif cmd == "snap":
        snap()
    elif cmd == "kill":
        kill()
    elif cmd == "focus":
        focus(sys.argv[2])
    else:
        raise SystemExit(f"未知命令: {cmd}")
