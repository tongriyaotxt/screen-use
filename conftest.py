import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture(autouse=True)
def _memory_home(tmp_path, monkeypatch):
    """所有测试的记忆存储指向临时目录，不污染真实环境。"""
    monkeypatch.setenv("SCREEN_USE_HOME", str(tmp_path / "screen_use_home"))
