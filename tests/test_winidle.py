"""用户空闲闸门测试：默认保守（取不到信息就当人在用），等待逻辑可注入。"""

import time

from screen_use import winidle


def test_user_idle_seconds_returns_number():
    v = winidle.user_idle_seconds()
    assert isinstance(v, float)
    assert v >= 0.0


def test_wait_for_idle_returns_immediately_when_idle(monkeypatch):
    monkeypatch.setattr(winidle, "user_idle_seconds", lambda: 999.0)
    t0 = time.perf_counter()
    waited = winidle.wait_for_idle(30.0, poll_s=0.01)
    assert waited < 0.5
    assert time.perf_counter() - t0 < 0.5


def test_wait_for_idle_waits_until_user_leaves(monkeypatch):
    """人在用（idle=1s）→ 一直等；直到 idle 达到阈值才返回。"""
    seq = iter([1.0, 1.0, 1.0, 40.0])
    monkeypatch.setattr(winidle, "user_idle_seconds", lambda: next(seq, 40.0))
    calls = []
    waited = winidle.wait_for_idle(30.0, poll_s=0.01, on_wait=lambda w, i: calls.append(i))
    assert waited >= 0
    assert calls == [1.0, 1.0, 1.0]          # 每次等待都回调（供打日志）


def test_wait_for_idle_respects_max_wait(monkeypatch):
    monkeypatch.setattr(winidle, "user_idle_seconds", lambda: 0.5)
    t0 = time.perf_counter()
    waited = winidle.wait_for_idle(30.0, poll_s=0.01, max_wait_s=0.05)
    assert waited < 1.0
    assert time.perf_counter() - t0 < 1.0    # 有上限就不会卡住


def test_user_idle_seconds_failure_is_conservative(monkeypatch):
    """API 异常 → 0.0（当作人在用），绝不能返回大值让自动化误判"人走了"。"""
    class _Boom:
        def __getattr__(self, name):
            raise OSError("no user32")

    monkeypatch.setattr(winidle.ctypes, "windll", _Boom())
    assert winidle.user_idle_seconds() == 0.0
