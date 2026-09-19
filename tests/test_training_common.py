"""训练侧数据通路的标签正确性（`training.common.iter_samples`）。

2026-09-19 实测 bug：docstring 写着「跳过…执行失败步」，代码里却没有这个判断。
代价在 mock v3 上显形——290 步里有 14 步是「oracle 目标不可见」的失败点击，
`args` 为空 → `elem_slot` 落到 no-element 槽 → 训练样本变成
「goal: 在列表最底部选中最后一项 → CLICK 无目标元素」，等于教模型点空气。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common import iter_samples  # noqa: E402

ELEMENTS = [
    {"id": 3, "control_type": "ButtonControl", "name": "删除", "bbox": [0, 0, 10, 10]},
    {"id": 7, "control_type": "EditControl", "name": "文件名", "bbox": [0, 20, 10, 10]},
]


def _step(action, args, ok=True, detail="", elements=None, goal="把改动存下来"):
    return {
        "goal": goal, "step_idx": 1, "foreground_title": "MockUI",
        "elements": ELEMENTS if elements is None else elements,
        "action": action, "args": args, "ok": ok, "detail": detail,
        "source": "mock_oracle", "quality": "gold", "ts": 0.0,
    }


def _write(tmp_path: Path, steps: list[dict]) -> Path:
    p = tmp_path / "ep.jsonl"
    p.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in steps),
                 encoding="utf-8")
    return p


def test_failed_steps_do_not_become_samples(tmp_path):
    """失败步（oracle 目标不可见）绝不能被当成「CLICK 无目标」的正样本。"""
    _write(tmp_path, [
        _step("click_id", {"id": 3}),                     # 正常步
        _step("click_id", {}, ok=False, detail="oracle 目标不可见: 项目10"),
        _step("done", {"result": "已保存"}),
    ])

    samples = iter_samples([tmp_path])

    assert [s["op"] for s in samples] == ["CLICK", "DONE"]
    assert all(s["ok"] for s in samples)
    # 失败步不能留下"无目标 CLICK"的标签
    assert not [s for s in samples if s["op"] == "CLICK" and s["elem_slot"] == s["n_elements"]]


def test_failed_step_still_shows_up_in_history(tmp_path):
    """失败动作要从 history 里看得见，后面步的运行历史才忠实。"""
    _write(tmp_path, [
        _step("click_id", {"id": 3}),
        _step("click_id", {}, ok=False, detail="oracle 目标不可见: 项目10"),
        _step("done", {"result": "已保存"}),
    ])

    samples = iter_samples([tmp_path])
    done = next(s for s in samples if s["op"] == "DONE")

    assert len(done["history"]) == 2, done["history"]
    assert "✗" in done["history"][-1], done["history"]


def test_target_not_in_observation_is_dropped(tmp_path):
    """标签指向的元素不在观测里 → 脏数据，丢样本但保留 history（原有行为不能丢）。"""
    _write(tmp_path, [
        _step("click_id", {"id": 999}),   # 观测里没有 999
        _step("done", {"result": "完成"}),
    ])

    samples = iter_samples([tmp_path])

    assert [s["op"] for s in samples] == ["DONE"]
    assert len(samples[0]["history"]) == 1


def test_scroll_uses_the_no_element_slot(tmp_path):
    """没有目标元素的动作（SCROLL/DONE/BLOCKED）才是 no-element 槽的合法主人。"""
    _write(tmp_path, [
        _step("scroll", {"clicks": -3}),
        _step("blocked", {"reason": "找不到目标"}),
        _step("done", {"result": "完成"}),
    ])

    samples = iter_samples([tmp_path])

    assert [s["op"] for s in samples] == ["SCROLL", "BLOCKED", "DONE"]
    assert all(s["elem_slot"] == s["n_elements"] for s in samples)


def test_type_step_gets_goal_copy_span(tmp_path):
    """TYPE 的文本要从 goal 里定位（span 头）；定位不到就给 None，交给 span loss mask。"""
    _write(tmp_path, [
        _step("type", {"text": "报告", "element_id": 7}, goal="在文件名里输入报告"),
        _step("type", {"text": "没出现在目标里的词", "element_id": 7}, goal="在文件名里输入报告"),
    ])

    samples = iter_samples([tmp_path])
    typed = [s for s in samples if s["op"] == "TYPE"]

    assert len(typed) == 2, "定位失败不该丢样本，只是没有 span 可监督"
    assert typed[0]["span"] == [7, 9]      # goal 里「报告」的字符区间
    assert typed[1]["span"] is None       # 定位不到 → mask
    assert typed[0]["elem_slot"] == 1     # 元素 7 在 ELEMENTS 里的下标
