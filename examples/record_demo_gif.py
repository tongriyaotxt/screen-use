"""录制 README 演示 GIF：自动操作计算器 123 + 456 = 579，同步录屏。

运行：python examples/record_demo_gif.py  →  assets/demo.gif
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mss
import uiautomation as auto
from PIL import Image

from screen_use.tools import ScreenUse

auto.SetGlobalSearchTimeout(5)

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "demo.gif")
FPS = 5
GIF_WIDTH = 480


def main() -> None:
    os.startfile("calc.exe")
    time.sleep(2)
    win = auto.WindowControl(searchDepth=1, SubName="Calculator")
    win.SetActive()
    time.sleep(0.5)
    r = win.BoundingRectangle
    region = {"left": r.left, "top": r.top, "width": r.width(), "height": r.height()}

    frames: list[Image.Image] = []
    stop = threading.Event()

    def capture() -> None:
        while not stop.is_set():
            with mss.MSS() as sct:
                raw = sct.grab(region)
            frames.append(Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX"))
            time.sleep(1 / FPS)

    t = threading.Thread(target=capture, daemon=True)
    t.start()
    time.sleep(0.5)  # 录一小段静止画面做片头

    tools = ScreenUse()
    for step in ["Clear", "One", "Two", "Three", "Plus",
                 "Four", "Five", "Six", "Equals"]:
        tools.click_element(step)
        time.sleep(0.4)

    time.sleep(1.0)  # 片尾停留，让观众看清结果
    stop.set()
    t.join()

    # 缩放并保存 GIF
    ratio = GIF_WIDTH / frames[0].size[0]
    size = (GIF_WIDTH, int(frames[0].size[1] * ratio))
    resized = [f.resize(size, Image.LANCZOS) for f in frames]
    resized[0].save(
        OUT, save_all=True, append_images=resized[1:],
        duration=int(1000 / FPS), loop=0, optimize=True,
    )
    print(f"saved {OUT}: {len(frames)} frames, {os.path.getsize(OUT)//1024} KB")


if __name__ == "__main__":
    main()
