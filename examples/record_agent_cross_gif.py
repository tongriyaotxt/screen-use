"""录制跨应用自主操作 GIF（简化版）：全屏 + 底部 AI 思考状态栏。

设计思路：
- 计算部分由脚本预置（确定性），GIF 聚焦跨应用能力
- Agent 任务：读计算器结果 → 切记事本 → 输入（3-4 步决策链）
- 多窗口枚举 + 窗口激活修复后，非前台窗口的元素也能直接点击

运行：python examples/record_agent_cross_gif.py  →  assets/demo_agent_cross.gif
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mss
import uiautomation as auto
from PIL import Image, ImageDraw, ImageFont

from screen_use import ScreenUse

auto.SetGlobalSearchTimeout(5)

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "demo_agent_cross.gif")
SCRATCH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", f"scratch_cross_{int(time.time())}.txt"
))
GIF_W = 960
BAR_RATIO = 0.22          # 更高状态栏，展示更多思考文本
FRAME_MS = 400            # 更快节奏
BG = (18, 18, 26)
GREEN = (80, 220, 120)
CYAN = (90, 200, 250)
GRAY = (170, 170, 185)

FONT_PATHS = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
F_GOAL = None
F_STEP = None

_state = {"events": []}
GOAL = "把计算器显示屏上的结果数字输入到记事本窗口里"


def load_font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def find_notepad():
    for w in auto.GetRootControl().GetChildren():
        try:
            if w.ClassName == "Notepad" and "scratch_cross" in (w.Name or ""):
                return w
        except Exception:
            pass
    return None


def wrap(text, width):
    return [text[i:i + width] for i in range(0, len(text), width)] or [""]


def render_frame(sct) -> Image.Image:
    raw = sct.grab(sct.monitors[0])
    img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    img = img.resize((GIF_W, int(GIF_W * img.height / img.width)), Image.LANCZOS)

    bar_h = int(img.height * BAR_RATIO)
    canvas = Image.new("RGB", (GIF_W, img.height + bar_h), BG)
    canvas.paste(img, (0, 0))

    d = ImageDraw.Draw(canvas)
    y0 = img.height
    d.rectangle([0, y0, GIF_W, canvas.height], fill=(10, 14, 24))
    d.line([(0, y0), (GIF_W, y0)], fill=(0, 220, 200), width=2)
    d.text((12, y0 + 5), "SCREEN-USE :: AGENT LOOP", font=F_GOAL, fill=(0, 220, 200))
    d.text((300, y0 + 5), f"GOAL> {GOAL}", font=F_GOAL, fill=CYAN)
    y = y0 + 32
    for tag, text, color in _state["events"][-4:]:
        for line in wrap(f"{tag} {text}", 92):
            if y > canvas.height - 20:
                break
            d.text((12, y), line, font=F_STEP, fill=color)
            y += 20
    return canvas


def main() -> None:
    global F_GOAL, F_STEP
    F_GOAL = load_font(16)
    F_STEP = load_font(14)

    tools = ScreenUse()

    # 预清理残留标签页
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

    # 1. 记事本（空草稿）放左上
    open(SCRATCH, "w").write("")
    os.startfile(SCRATCH)
    time.sleep(3)
    np_win = find_notepad()
    assert np_win, "notepad window not found"
    np_win.GetTransformPattern().Move(20, 20)
    time.sleep(0.3)

    # 2. 计算器放右侧，脚本预置算好 56×7=392（确定性操作）
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
    print("环境就绪：计算器=392，记事本空", flush=True)

    # 3. 录制 + Agent 自主完成跨应用转移
    frames: list[Image.Image] = []

    def snapshot(hold=1):
        with mss.MSS() as sct:
            frames.extend([render_frame(sct)] * hold)

    _state["events"].append(("[SYS ]", "run_task() 启动，VLM 开始观察屏幕...", GRAY))
    snapshot(2)

    def on_step(step):
        icon = "+" if step.ok else "x"
        if step.thought:
            _state["events"].append(("[THINK]", step.thought, GRAY))
        _state["events"].append(
            (f"[ACT {icon}]", f"step{step.index}: {step.action} {step.args}",
             GREEN if step.ok else CYAN)
        )
        snapshot(1)

    result = tools.run_task(GOAL, max_steps=10, on_step=on_step)

    _state["events"].append(("[DONE ]", str(result["result"]), CYAN))
    snapshot(4)

    resized = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    resized[0].save(
        OUT, save_all=True, append_images=resized[1:],
        duration=FRAME_MS, loop=0, optimize=True,
    )
    print(f"saved {OUT}: {len(frames)} frames, {os.path.getsize(OUT)//1024} KB")
    print(f"done={result['done']} steps={len(result['steps'])} result={result['result']}")

    # 校验 + 清理
    win = find_notepad()
    if win:
        doc = win.DocumentControl(searchDepth=10)
        content = doc.GetTextPattern().DocumentRange.GetText(-1)
        print("notepad content:", repr(content))
        assert "392" in content, f"结果未写入记事本: {content!r}"
        win.GetWindowPattern().Close()
        time.sleep(1)
        r2 = tools.click_element("Don't save")
        if not r2.get("found"):
            tools.click_element("不保存")
    if os.path.exists(SCRATCH):
        os.remove(SCRATCH)
    print("✅ 跨应用自主操作验证通过: 392 已写入记事本")


if __name__ == "__main__":
    main()
