"""内省系统测试：困难分类 + 提示词生成。"""

from screen_use.agent import Step
from screen_use.reflect import (
    StuckSignal,
    StuckType,
    build_reflect_prompt,
    classify_stuck,
    screen_fingerprint,
)


def _step(action, args=None, ok=True):
    return Step(index=1, thought="", action=action, args=args or {}, ok=ok)


def _signal(steps, changed=True, elements=10, title_changed=False):
    return StuckSignal(
        steps=steps,
        screen_changed=changed,
        elements_count=elements,
        window_title_changed=title_changed,
    )


def test_not_stuck():
    sig = _signal([_step("click_id", {"id": 1})], changed=True)
    assert classify_stuck(sig) is None


def test_repeat_loop():
    steps = [_step("click", {"x": 1, "y": 2}) for _ in range(3)]
    assert classify_stuck(_signal(steps)) == StuckType.REPEAT_LOOP


def test_consecutive_failures():
    steps = [_step("click_id", {"id": 9}, ok=False), _step("click_id", {"id": 8}, ok=False)]
    assert classify_stuck(_signal(steps)) == StuckType.CONSECUTIVE_FAILURES


def test_no_effect():
    # 上一步成功，但屏幕没变化（NO_EFFECT 检查 steps[-2]，即产生当前截图的那步）
    steps = [_step("click", {"x": 1, "y": 1}), _step("click", {"x": 3, "y": 4})]
    assert classify_stuck(_signal(steps, changed=False)) == StuckType.NO_EFFECT


def test_no_effect_ignores_wait():
    steps = [_step("click", {"x": 1, "y": 1}), _step("wait", {"seconds": 1})]
    # steps[-2] 是 click 且屏幕变了 → 不卡住
    assert classify_stuck(_signal(steps, changed=True)) is None


def test_no_elements():
    steps = [_step("click", {"x": 1, "y": 1})]
    assert classify_stuck(_signal(steps, elements=0)) == StuckType.NO_ELEMENTS


def test_popup():
    steps = [_step("click", {"x": 1, "y": 1})]
    assert classify_stuck(_signal(steps, title_changed=True)) == StuckType.POPUP


def test_priority_repeat_over_failures():
    # 三次重复且都失败 → 重复循环优先
    steps = [_step("click", {"x": 1, "y": 1}, ok=False) for _ in range(3)]
    assert classify_stuck(_signal(steps)) == StuckType.REPEAT_LOOP


def test_fingerprint_stable():
    assert screen_fingerprint("abc") == screen_fingerprint("abc")
    assert screen_fingerprint("abc") != screen_fingerprint("abd")


def test_build_prompt_contains_guidance():
    prompt = build_reflect_prompt(StuckType.REPEAT_LOOP, "测试任务", ["第1步: click"])
    assert "重复循环" in prompt and "禁止再执行相同的动作" in prompt
    assert "测试任务" in prompt and "第1步" in prompt


def test_every_type_has_template():
    for t in StuckType:
        prompt = build_reflect_prompt(t, "g", [])
        assert "analysis" in prompt and "strategy" in prompt
