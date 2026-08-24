"""录制 Agent 视觉循环演示 GIF（全屏版）：整屏画面 + 底部 AI 思考状态栏。

每一步点击都是 VLM 看截图后自主决策的。
运行：python examples/record_agent_loop_gif.py  →  assets/demo_agent.gif
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

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "demo_agent.gif")
GIF_W = 960
BAR_RATIO = 0.14
BG = (18, 18, 26)
GREEN = (80, 220, 120)
CYAN = (90, 200, 250)
GRAY = (170, 170, 185)

FONT_PATHS = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
F_GOAL = None
F_STEP = None

_state = {"events": []}
GOAL = "点 7、8、乘号、9、等号，算出结果"


def load_font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def wrap(text, width):
    return [text[i:i + width] for i in range(0, len(text), width)] or [""]


def render_frame(sct) -> Image.Image:
    """全屏截图 + 底部状态栏叠加。"""
    raw = sct.grab(sct.monitors[0])
    img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    img = img.resize((GIF_W, int(GIF_W * img.height / img.width)), Image.LANCZOS)

    bar_h = int(img.height * BAR_RATIO)
    canvas = Image.new("RGB", (GIF_W, img.height + bar_h), BG)
    canvas.paste(img, (0, 0))

    d = ImageDraw.Draw(canvas)
    y0 = img.height
    d.rectangle([0, y0, GIF_W, canvas.height], fill=(24, 24, 36))
    d.text((12, y0 + 6), f"goal: {GOAL}", font=F_GOAL, fill=CYAN)
    y = y0 + 34
    for text, color in _state["events"][-2:]:
        for line in wrap(text, 88):
            if y > canvas.height - 22:
                break
            d.text((12, y), line, font=F_STEP, fill=color)
            y += 21
    return canvas


def main() -> None:
    global F_GOAL, F_STEP
    F_GOAL = load_font(16)
    F_STEP = load_font(14)

    tools = ScreenUse()

    # 打开计算器并激活（关键：首次成功的前置条件）
    os.startfile("calc.exe")
    time.sleep(2.5)
    calc = auto.WindowControl(searchDepth=1, SubName="Calculator")
    calc.SetActive()
    time.sleep(0.5)
    tools.click_element("Clear")

    frames: list[Image.Image] = []

    def snapshot(hold=1):
        with mss.MSS() as sct:
            frame = render_frame(sct)
        frames.extend([frame] * hold)

    _state["events"].append(("> run_task() 启动，VLM 开始观察屏幕...", GRAY))
    snapshot(2)

    def on_step(step):
        icon = "+" if step.ok else "x"
        if step.thought:
            _state["events"].append((f"  {step.thought}", GRAY))
        _state["events"].append(
            (f"[{icon}] step{step.index}: {step.action} {step.args}", GREEN if step.ok else CYAN)
        )
        snapshot(1)

    result = tools.run_task(GOAL, max_steps=10, on_step=on_step)

    _state["events"].append((f">>> done: {result['result']}", CYAN))
    snapshot(3)

    resized = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    resized[0].save(
        OUT, save_all=True, append_images=resized[1:],
        duration=900, loop=0, optimize=True,
    )
    print(f"saved {OUT}: {len(frames)} frames, {os.path.getsize(OUT)//1024} KB")
    print(f"done={result['done']} steps={len(result['steps'])} result={result['result']}")


if __name__ == "__main__":
    main()
