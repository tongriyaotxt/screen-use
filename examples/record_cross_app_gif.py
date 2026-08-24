"""录制跨应用演示 GIF：计算器算 123+456 → 结果粘贴进记事本。

画面合成在纯色画布上（只截取目标窗口区域），不泄露桌面其他内容。
运行：python examples/record_cross_app_gif.py  →  assets/demo_cross_app.gif
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

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "demo_cross_app.gif")
SCRATCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scratch_demo.txt"))
FPS = 4
GIF_WIDTH = 800

# 画布布局（物理像素）
CANVAS = (980, 560)
NOTEPAD_POS = (20, 20, 560, 500)   # x, y, w, h
CALC_POS = (620, 20, 340, 520)
BG = (24, 24, 32)

_state = {"calc_rect": None, "np_rect": None}


def find_notepad():
    for w in auto.GetRootControl().GetChildren():
        try:
            if w.ClassName == "Notepad" and "scratch_demo" in w.Name:
                return w
        except Exception:
            pass
    return None


def grab(sct, rect):
    raw = sct.grab({"left": rect[0], "top": rect[1], "width": rect[2], "height": rect[3]})
    return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")


def main() -> None:
    tools = ScreenUse()

    # 1. 准备记事本（隔离草稿文件），并摆位
    with open(SCRATCH, "w", encoding="ascii") as f:
        f.write("The answer is: ")
    os.startfile(SCRATCH)
    time.sleep(3)
    np_win = find_notepad()
    assert np_win, "notepad window not found"
    np_win.SetActive()
    time.sleep(0.5)
    tp = np_win.GetTransformPattern()
    tp.Move(NOTEPAD_POS[0], NOTEPAD_POS[1])
    time.sleep(0.5)
    nr = np_win.BoundingRectangle
    _state["np_rect"] = (nr.left, nr.top, nr.width(), nr.height())

    # 光标移到文本末尾
    doc = np_win.DocumentControl(searchDepth=10)
    tools.click(NOTEPAD_POS[0] + 200, NOTEPAD_POS[1] + 120)
    tools.hotkey("ctrl", "end")
    time.sleep(0.3)

    # 2. 录制线程：双窗口合成到纯色画布
    frames: list[Image.Image] = []
    stop = threading.Event()

    def capture() -> None:
        with mss.MSS() as sct:
            while not stop.is_set():
                canvas = Image.new("RGB", CANVAS, BG)
                if _state["np_rect"]:
                    canvas.paste(grab(sct, _state["np_rect"]), (_state["np_rect"][0], _state["np_rect"][1]))
                if _state["calc_rect"]:
                    canvas.paste(grab(sct, _state["calc_rect"]), (_state["calc_rect"][0], _state["calc_rect"][1]))
                frames.append(canvas)
                time.sleep(1 / FPS)

    t = threading.Thread(target=capture, daemon=True)
    t.start()
    time.sleep(0.8)

    # 3. 打开计算器，摆位，算 123 + 456
    os.startfile("calc.exe")
    time.sleep(2.5)
    calc = auto.WindowControl(searchDepth=1, SubName="Calculator")
    calc.SetActive()
    time.sleep(0.5)
    ctp = calc.GetTransformPattern()
    ctp.Move(CALC_POS[0], CALC_POS[1])
    time.sleep(0.5)
    cr = calc.BoundingRectangle
    _state["calc_rect"] = (cr.left, cr.top, cr.width(), cr.height())

    for step in ["Clear", "One", "Two", "Three", "Plus",
                 "Four", "Five", "Six", "Equals"]:
        tools.click_element(step)
        time.sleep(0.35)

    # 4. 复制结果
    time.sleep(0.5)
    tools.hotkey("ctrl", "c")
    time.sleep(0.5)

    # 5. 切回记事本，粘贴
    np_win.SetActive()
    time.sleep(0.8)
    tools.click(NOTEPAD_POS[0] + 200, NOTEPAD_POS[1] + 120)
    tools.hotkey("ctrl", "end")
    tools.hotkey("ctrl", "v")
    time.sleep(1.5)

    stop.set()
    t.join()

    # 6. 校验结果
    doc = find_notepad().DocumentControl(searchDepth=10)
    content = doc.GetTextPattern().DocumentRange.GetText(-1)
    print("notepad content:", repr(content))
    assert "579" in content, "跨应用粘贴失败"

    # 7. 保存 GIF
    ratio = GIF_WIDTH / CANVAS[0]
    size = (GIF_WIDTH, int(CANVAS[1] * ratio))
    resized = [f.resize(size, Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    resized[0].save(
        OUT, save_all=True, append_images=resized[1:],
        duration=int(1000 / FPS), loop=0, optimize=True,
    )
    print(f"saved {OUT}: {len(frames)} frames, {os.path.getsize(OUT)//1024} KB")

    # 8. 清理
    find_notepad().GetWindowPattern().Close()
    time.sleep(1)
    if os.path.exists(SCRATCH):
        os.remove(SCRATCH)
    print("✅ 跨应用验证通过: 123 + 456 = 579 已粘贴进记事本")


if __name__ == "__main__":
    main()
