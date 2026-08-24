"""元学习记忆存储测试。"""

import json

from screen_use.memory import ExperienceStore, _tokenize


def _store(tmp_path) -> ExperienceStore:
    return ExperienceStore(tmp_path / "mem")


def test_record_and_recall(tmp_path):
    store = _store(tmp_path)
    result = {
        "done": True, "result": "ok",
        "steps": [{"action": "click_id", "args": {"id": 1}, "ok": True}],
    }
    store.record_trace("打开计算器算加法", result)
    store.record_trace("在记事本里写日记", result)

    hits = store.recall("用计算器做加法")
    assert len(hits) >= 1
    assert "计算器" in hits[0]["goal"]


def test_recall_prefers_success(tmp_path):
    store = _store(tmp_path)
    fail = {"done": False, "result": "x", "steps": []}
    ok = {"done": True, "result": "x", "steps": []}
    store.record_trace("计算器 任务A", fail)
    store.record_trace("计算器 任务B", ok)
    hits = store.recall("计算器 任务")
    assert hits[0]["goal"].endswith("B")  # 成功的排前面


def test_recall_no_overlap(tmp_path):
    store = _store(tmp_path)
    store.record_trace("打开计算器", {"done": True, "result": "", "steps": []})
    assert store.recall("zzzzz") == []


def test_vocab_learn_and_recall(tmp_path):
    store = _store(tmp_path)
    store.learn_mapping("乘号", "Multiply by", "ButtonControl")
    hit = store.recall_mapping("乘号")
    assert hit["name"] == "Multiply by"
    assert hit["hits"] == 1
    store.learn_mapping("乘号", "Multiply by", "ButtonControl")
    assert store.recall_mapping("乘号")["hits"] == 2
    assert store.recall_mapping("不存在") is None


def test_stats(tmp_path):
    store = _store(tmp_path)
    r = {"done": True, "result": "", "steps": []}
    store.record_trace("任务1", r)
    store.record_trace("任务2", {**r, "done": False})
    store.learn_mapping("保存", "Save")
    s = store.stats()
    assert s["traces"] == 2 and s["successes"] == 1 and s["vocab_size"] == 1


def test_tokenize_mixed():
    tokens = _tokenize("打开 Calculator 窗口")
    assert "calculator" in tokens and "打" in tokens and "窗" in tokens
