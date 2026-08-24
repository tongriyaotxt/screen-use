"""屏幕截图：mss 采集 + DPI 适配。

进程启动时设为 DPI-aware，保证 mss 截图坐标、UIA 坐标、pyautogui 点击坐标
三者统一为物理像素，避免 Windows 缩放（125%/150%）导致的坐标错位。
"""

from __future__ import annotations

import base64
import ctypes
import io

import mss
from PIL import Image

_dpi_aware_done = False


def ensure_dpi_aware() -> None:
    """将进程标记为 DPI-aware（只需调用一次）。"""
    global _dpi_aware_done
    if _dpi_aware_done:
        return
    try:
        # Windows 8.1+，PROCESS_PER_MONITOR_DPI_AWARE = 2
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass
    _dpi_aware_done = True


def screenshot(monitor: int = 0) -> Image.Image:
    """截取屏幕，返回 PIL Image（物理像素）。monitor=0 表示全部显示器拼接。"""
    ensure_dpi_aware()
    with mss.MSS() as sct:
        raw = sct.grab(sct.monitors[monitor])
        return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")


def downscale(img: Image.Image, max_size: int) -> tuple[Image.Image, float]:
    """等比缩放至最长边不超过 max_size。返回 (缩放后图片, 缩放比例)。

    缩放比例 scale = 原图 / 新图，用于把缩放图上的坐标换算回真实屏幕坐标。
    max_size <= 0 时不缩放。
    """
    if max_size <= 0:
        return img, 1.0
    w, h = img.size
    longest = max(w, h)
    if longest <= max_size:
        return img, 1.0
    ratio = max_size / longest
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return img.resize(new_size, Image.LANCZOS), longest / max_size


def to_jpeg_base64(img: Image.Image, quality: int = 80) -> str:
    """编码为 JPEG base64，用于 MCP 传输和 VLM 调用。"""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def capture_for_model(max_size: int, quality: int = 80) -> tuple[Image.Image, str, float]:
    """截图 → 缩放 → 编码，一步到位。返回 (缩放后图, base64, 坐标换算比例)。"""
    img = screenshot()
    scaled, scale = downscale(img, max_size)
    return scaled, to_jpeg_base64(scaled, quality), scale
