"""用户空闲检测：判断「人现在在不在用这台电脑」——零干扰自动化的闸门。

`GETLASTINPUTINFO` 返回的是**真实用户**的输入时间戳；我们用 UIA Invoke / ValuePattern
做的操作不产生输入事件，所以这个值不会被自己的动作污染（已由 mock v2 的零干扰设计保证）。

用途：owner 说「可以跑，但别影响我用电脑」——量产/训练每步前问一次，
人在用就暂停、人离开就继续（见 `scripts/mock_v3.py`、`training/train.py` 的 --idle-gate-s）。

取不到信息时一律返回 0（当作"人在用"）：宁可慢，也不要偷偷抢占用户机器。
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


def user_idle_seconds() -> float:
    """距上一次键盘/鼠标输入过了多少秒（取不到 → 0.0）。"""
    try:
        info = _LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(_LASTINPUTINFO)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return 0.0
        now = ctypes.windll.kernel32.GetTickCount()
        # dwTime 是 32 位毫秒计数，约 49.7 天回绕一次；用掩码做差抗回绕
        elapsed_ms = (now - info.dwTime) & 0xFFFFFFFF
        return elapsed_ms / 1000.0
    except Exception:  # noqa: BLE001 - 非 Windows / API 异常都按"人在用"处理
        return 0.0


def wait_for_idle(idle_s: float, poll_s: float = 5.0,
                  on_wait=None, max_wait_s: float = 0.0) -> float:
    """等到用户空闲满 idle_s 秒（或超过 max_wait_s 上限）。返回实际等待秒数。

    max_wait_s=0 表示不设上限（一直等）——默认行为是"人不用了才继续"。
    on_wait: 可选回调 fn(waited_s, idle_s)，用来打日志（每 poll_s 一次）。
    """
    t0 = time.perf_counter()
    while True:
        idle = user_idle_seconds()
        if idle >= idle_s:
            return time.perf_counter() - t0
        waited = time.perf_counter() - t0
        if max_wait_s and waited >= max_wait_s:
            return waited
        if on_wait:
            on_wait(waited, idle)
        time.sleep(poll_s)
