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
_sct = None  # 模块级缓存的 mss 实例（避免每次截图新建，mss 非线程安全但本工具单线程使用）


def _get_sct():
    """获取复用的 mss 实例（懒创建）。"""
    global _sct
    if _sct is None:
        _sct = mss.MSS()
    return _sct


def _reset_sct() -> None:
    """销毁缓存的 mss 实例。

    mss 的 Windows GDI 后端在 MSS() 初始化时创建 srcdc/memdc（GetWindowDC/
    CreateCompatibleDC）并反复复用；显示器睡眠唤醒、驱动重置、分辨率/DPI 变更、
    RDP 会话切换等都会让这些 DC 句柄失效，之后每次 BitBlt 都失败。重建实例
    即可拿到新 DC，无需重启进程。
    """
    global _sct
    if _sct is not None:
        try:
            _sct.close()
        except Exception:  # noqa: BLE001 - close 失败不影响后续重建
            pass
        _sct = None


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
    """截取屏幕，返回 PIL Image（物理像素）。monitor=0 表示全部显示器拼接。

    GDI 的 BitBlt 在 DC 句柄失效后会持续报错（见 _reset_sct 注释），
    捕获 ScreenShotError 后重建 mss 实例重试一次，避免 MCP 服务永久截不了图。
    """
    ensure_dpi_aware()
    sct = _get_sct()
    try:
        raw = sct.grab(sct.monitors[monitor])
    except mss.exception.ScreenShotError:
        _reset_sct()
        sct = _get_sct()
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
    return img.resize(new_size, Image.BILINEAR), longest / max_size


def to_jpeg_base64(img: Image.Image, quality: int = 80) -> str:
    """编码为 JPEG base64，用于 MCP 传输和 VLM 调用。"""
    return base64.b64encode(to_jpeg_bytes(img, quality)).decode("ascii")


def to_jpeg_bytes(img: Image.Image, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def to_jpeg_base64_capped(
    img: Image.Image,
    quality: int = 80,
    max_bytes: int = 90 * 1024,
    fallback_qualities: tuple[int, ...] = (60, 40),
) -> tuple[str, int]:
    """编码为 JPEG base64，字节数超过 max_bytes 时逐步降 quality 重编码。

    返回 (base64, 实际使用的 quality)。保证 MCP 输出不被宿主按大小截断。
    """
    data = to_jpeg_bytes(img, quality)
    used = quality
    for q in fallback_qualities:
        if len(data) <= max_bytes:
            break
        data = to_jpeg_bytes(img, q)
        used = q
    return base64.b64encode(data).decode("ascii"), used


def capture_for_model(max_size: int, quality: int = 80) -> tuple[Image.Image, str, float]:
    """截图 → 缩放 → 编码，一步到位。返回 (缩放后图, base64, 坐标换算比例)。"""
    img = screenshot()
    scaled, scale = downscale(img, max_size)
    return scaled, to_jpeg_base64(scaled, quality), scale
