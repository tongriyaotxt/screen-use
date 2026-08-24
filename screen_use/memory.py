"""元学习：跨任务经验记忆。

- 执行轨迹持久化（JSONL）：任务目标、步骤、成败，供后续任务检索参考
- 词汇映射学习：用户描述 ↔ 实际控件名（如 "乘号" ↔ "Multiply by"），
  成功后自动记录，下次直接命中策略链第 0 级

存储位置：~/.screen_use/（可用 SCREEN_USE_HOME 环境变量覆盖）
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path


def _default_home() -> Path:
    return Path(os.environ.get("SCREEN_USE_HOME", Path.home() / ".screen_use"))


def _tokenize(text: str) -> set[str]:
    """中英混合分词：英文按词，中文按字（简单够用）。"""
    text = text.lower()
    words = set(re.findall(r"[a-z0-9]+", text))
    chars = set(re.findall(r"[一-鿿]", text))
    return words | chars


class ExperienceStore:
    def __init__(self, home: Path | None = None):
        self.home = home or _default_home()
        self.home.mkdir(parents=True, exist_ok=True)
        self.traces_path = self.home / "traces.jsonl"
        self.vocab_path = self.home / "vocab.json"

    # ---- 执行轨迹 ----

    def record_trace(self, goal: str, result: dict) -> None:
        """任务结束后记录轨迹（只存摘要，不存截图）。"""
        entry = {
            "ts": time.time(),
            "goal": goal,
            "done": result["done"],
            "result": result["result"],
            "steps": [
                {"action": s["action"], "args": s["args"], "ok": s["ok"]}
                for s in result["steps"]
            ],
        }
        with open(self.traces_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def recall(self, goal: str, limit: int = 3) -> list[dict]:
        """按关键词重叠度检索相似历史任务（简单有效，零依赖）。"""
        if not self.traces_path.exists():
            return []
        query = _tokenize(goal)
        if not query:
            return []
        scored = []
        with open(self.traces_path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                overlap = len(query & _tokenize(entry.get("goal", "")))
                if overlap > 0:
                    # 成功的经验优先，其次按重叠度
                    scored.append((entry.get("done", False), overlap, entry))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [e for _, _, e in scored[:limit]]

    # ---- 词汇映射 ----

    def _load_vocab(self) -> dict:
        if not self.vocab_path.exists():
            return {}
        try:
            return json.loads(self.vocab_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def learn_mapping(self, description: str, element_name: str, control_type: str = "") -> None:
        """记录 描述→控件名 的成功映射（重复成功会累计置信度）。"""
        key = description.strip().lower()
        if not key or not element_name:
            return
        vocab = self._load_vocab()
        entry = vocab.get(key, {"name": "", "control_type": "", "hits": 0})
        entry["name"] = element_name
        if control_type:
            entry["control_type"] = control_type
        entry["hits"] = entry.get("hits", 0) + 1
        vocab[key] = entry
        self.vocab_path.write_text(
            json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def recall_mapping(self, description: str) -> dict | None:
        """查找已学到的映射。返回 {"name": ..., "control_type": ..., "hits": ...}。"""
        return self._load_vocab().get(description.strip().lower())

    # ---- 统计 ----

    def stats(self) -> dict:
        traces = 0
        successes = 0
        if self.traces_path.exists():
            with open(self.traces_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        traces += 1
                        successes += bool(entry.get("done"))
                    except json.JSONDecodeError:
                        continue
        return {
            "traces": traces,
            "successes": successes,
            "vocab_size": len(self._load_vocab()),
        }
