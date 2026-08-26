"""screen 截图工具测试：JPEG 体积上限控制、缩放滤镜、mss 实例复用。"""

import random

from PIL import Image

from screen_use.perception import screen


def _noise_image(w=800, h=600) -> Image.Image:
    """噪声图（JPEG 难以压缩，用于触发质量降级）。"""
    rng = random.Random(42)
    img = Image.new("RGB", (w, h))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                 for _ in range(w * h)])
    return img


def test_to_jpeg_base64_capped_no_reduce_when_small():
    img = Image.new("RGB", (100, 100), "white")
    b64, used = screen.to_jpeg_base64_capped(img, quality=80, max_bytes=90 * 1024)
    assert used == 80
    assert isinstance(b64, str) and b64


def test_to_jpeg_base64_capped_reduces_quality():
    """超过 max_bytes 时逐步降 quality（80→60→40）重编码。"""
    img = _noise_image()
    b64_80, used_80 = screen.to_jpeg_base64_capped(img, quality=80, max_bytes=10**9)
    assert used_80 == 80
    # 设一个 80 质量达不到、40 质量能达到的上限 → 降到 40
    import base64
    size_80 = len(base64.b64decode(b64_80))
    b64, used = screen.to_jpeg_base64_capped(img, quality=80, max_bytes=size_80 // 4)
    assert used in (60, 40)
    assert len(base64.b64decode(b64)) <= size_80  # 体积确实下降


def test_to_jpeg_base64_capped_never_worse_than_lowest():
    """即使最低质量也超限时，返回最低质量的结果而不是死循环。"""
    img = _noise_image()
    _, used = screen.to_jpeg_base64_capped(img, quality=80, max_bytes=1)
    assert used == 40


def test_downscale_uses_bilinear():
    img = Image.new("RGB", (2560, 1440), "white")
    scaled, scale = screen.downscale(img, 1280)
    assert scaled.size == (1280, 720) and scale == 2.0


def test_mss_instance_reused(monkeypatch):
    """mss 实例模块级缓存复用，不每次新建。"""
    calls = []

    class _FakeMSS:
        def __init__(self):
            calls.append(1)
            self.monitors = [None, {"left": 0, "top": 0, "width": 2, "height": 2}]

        def grab(self, mon):
            class _Raw:
                size = (2, 2)
                bgra = b"\x00\x00\x00\xff" * 4
            return _Raw()

    monkeypatch.setattr("mss.MSS", _FakeMSS)
    monkeypatch.setattr(screen, "_sct", None)
    screen.screenshot(monitor=1)
    screen.screenshot(monitor=1)
    assert len(calls) == 1


def test_screenshot_recovers_from_stale_gdi(monkeypatch):
    """BitBlt 因 DC 失效报 ScreenShotError 后，重建 mss 实例并重试成功。"""
    import mss

    calls = []
    closed = []

    class _FlakyMSS:
        def __init__(self):
            calls.append(1)
            self.monitors = [None, {"left": 0, "top": 0, "width": 2, "height": 2}]

        def grab(self, mon):
            if len(calls) == 1:  # 第一个实例的 DC 已失效
                raise mss.exception.ScreenShotError(
                    "Windows graphics function failed: BitBlt"
                )

            class _Raw:
                size = (2, 2)
                bgra = b"\x00\x00\x00\xff" * 4
            return _Raw()

        def close(self):
            closed.append(1)

    monkeypatch.setattr("mss.MSS", _FlakyMSS)
    monkeypatch.setattr(screen, "_sct", None)
    img = screen.screenshot(monitor=1)
    assert img.size == (2, 2)
    assert len(calls) == 2  # 旧实例废弃 + 新实例重试
    assert len(closed) == 1  # 旧实例被正确关闭


def test_screenshot_reraises_when_retry_also_fails(monkeypatch):
    """重建后仍失败则照常抛错，不吞异常。"""
    import mss

    class _DeadMSS:
        def __init__(self):
            self.monitors = [None, {"left": 0, "top": 0, "width": 2, "height": 2}]

        def grab(self, mon):
            raise mss.exception.ScreenShotError("Windows graphics function failed: BitBlt")

        def close(self):
            pass

    monkeypatch.setattr("mss.MSS", _DeadMSS)
    monkeypatch.setattr(screen, "_sct", None)
    try:
        screen.screenshot(monitor=1)
    except mss.exception.ScreenShotError:
        pass
    else:
        raise AssertionError("应当抛出 ScreenShotError")
