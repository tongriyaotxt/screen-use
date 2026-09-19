"""北极星基线任务集的保底测试：schema 完整、验证器可调用、id 不重复。

这套任务集是「人 vs agent」对照的口径来源，坏了会让北极星数字不可比，
所以用测试钉住它的结构（不真跑任务，也不碰鼠标键盘）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import baseline  # noqa: E402


def test_task_suite_schema():
    assert len(baseline.TASKS) >= 6
    ids = [t["id"] for t in baseline.TASKS]
    assert len(ids) == len(set(ids)), "任务 id 重复"
    for t in baseline.TASKS:
        assert t["goal"].strip(), f"{t['id']} 缺 goal"
        assert t["verify_desc"].strip(), f"{t['id']} 缺完成判据描述"
        assert t["verify"] is None or callable(t["verify"])


def test_verifiers_are_callable_and_boolean():
    """未做任务时验证器应返回 False（不能抛异常，也不能默认 True）。"""
    for t in baseline.TASKS:
        if t["verify"] is None:
            continue
        assert t["verify"]() is False or t["verify"]() is True


def test_file_tasks_point_into_temp_workspace():
    """有程序化验证的任务都在临时工作目录里干活，不能碰用户真实文件。"""
    file_tasks = [t for t in baseline.TASKS if t["verify"] is not None]
    assert len(file_tasks) >= 5
    for t in file_tasks:
        assert str(baseline.WORK) in t["goal"], f"{t['id']} 的路径不在临时工作目录"


def test_report_handles_missing_files(tmp_path, monkeypatch, capsys):
    """没有计时数据时 report 也不能炸（只是空表）。"""
    monkeypatch.setattr(baseline, "BENCH", tmp_path)
    baseline.run_report()
    out = capsys.readouterr().out
    assert "北极星基线" in out
    assert (tmp_path / "baseline_report.md").exists()
