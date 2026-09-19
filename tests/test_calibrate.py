"""校准分半与分布统计的护栏（2026-09-19 冒烟暴露的小样本崩溃）。

背景：`training.calibrate` 按轨迹文件哈希把留出集分半（拟合半 / 选择半）。
留出集只有 2 个文件时，哈希有可能把它们全分到同一侧，另一侧为空 →
`torch.cat([])` 抛 `ValueError: expected a non-empty list of Tensors`，
报错信息完全看不出是"数据太少"。这里锁住两条：
① 分半绝不把同一 episode 拆到两侧（防泄漏）；② 分不出两侧时要么按文件整体挪一个，
要么给出人能看懂的错误。
"""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.calibrate import _collect, split_calib_select, train_stats  # noqa: E402


def _names_on_one_side(n: int, even: bool) -> list[str]:
    """造 n 个哈希后落在同一侧的文件名（确定性搜索，不依赖运气）。"""
    out, i = [], 0
    while len(out) < n:
        name = f"ep-{i}.jsonl"
        hit = hashlib.sha1(name.encode("utf-8")).digest()[0] % 2 == 0
        if hit == even:
            out.append(name)
        i += 1
    return out


def _samples(files: list[str], per_file: int = 2) -> list[dict]:
    return [{"file": f} for f in files for _ in range(per_file)]


def _sides(samples: list[dict]) -> set[str]:
    return {s["file"] for s in samples}


def test_split_keeps_whole_files_on_one_side():
    """正常数据：同一 episode 的步只能出现在一侧（否则 ECE/覆盖率虚高）。"""
    files = [f"ep-{i}.jsonl" for i in range(12)]
    calib, select = split_calib_select(_samples(files))

    assert calib and select, "12 个文件不该分出一侧为空"
    assert not (_sides(calib) & _sides(select)), "同一文件不能跨两侧（泄漏）"
    assert _sides(calib) | _sides(select) == set(files), "文件不能丢"


def test_split_balances_when_hash_lands_on_one_side():
    """所有文件哈希到同一侧时，按文件整体挪一个过去，保证两侧都非空。"""
    files = _names_on_one_side(4, even=True)
    samples = _samples(files)
    assert {hashlib.sha1(f.encode()).digest()[0] % 2 for f in files} == {0}  # 前提成立

    calib, select = split_calib_select(samples)

    assert calib and select, "兜底没生效：仍有一侧为空"
    assert not (_sides(calib) & _sides(select)), "兜底也不能拆散同一文件"
    assert len(calib) + len(select) == len(samples), "样本不能丢"


def test_split_cannot_balance_a_single_file():
    """只有一个文件时无解——保持一侧为空，由 _collect 报出可读错误。"""
    calib, select = split_calib_select(_samples(["only.jsonl"]))

    assert not calib or not select
    assert len(calib) + len(select) == 2


def test_collect_rejects_empty_split_with_readable_error():
    """空校准集不能崩在 torch.cat 上，要说清是"数据太少"。"""
    with pytest.raises(SystemExit, match="校准集为空"):
        _collect(None, [], None, None)


def test_train_stats_reports_ood_gate_inputs():
    """train_stats 的 elements_max 是 serve 侧分布门控的阈值来源，必须有值。"""
    from training.common import serialize_observation

    seen: list[str] = []

    def fake_tok(text: str) -> dict:
        seen.append(text)
        return {"input_ids": list(range(len(text.split())))}

    elements = [{"id": i, "control_type": "ButtonControl", "name": f"按钮{i}",
                 "bbox": [0, 0, 10, 10]} for i in range(11)]
    samples = [{"goal": "把改动存下来", "foreground": "记事本", "elements": elements[:k],
                "history": []} for k in (3, 7, 11)]

    stats = train_stats(samples, fake_tok)

    assert stats["n"] == 3
    assert stats["elements_max"] == 11
    assert stats["elements_p50"] == 7
    assert stats["tokens_max"] > 0
    # 序列化用的是 train/serve 同一个函数
    assert seen[0] == serialize_observation(samples[0]["goal"], samples[0]["foreground"],
                                            samples[0]["elements"], samples[0]["history"])


def test_train_stats_empty_is_empty_dict():
    assert train_stats([], lambda t: {"input_ids": [1]}) == {}
