"""torch Dataset：样本 → token 序列 + 锚点位置 + 标签（TRAINING.md §2/§3）。

锚点定位方式（实现层关键决定）：
- 特殊 token 是词表里的独立 id，直接在 input_ids 里扫描定位：
  `<|elem|>` 出现 N 次 → 第 i 个位置即元素 i 的向量槽位；
  `<|decide|>` 位置 = 全局决策向量。
- goal-copy span：字符区间 → offset_mapping → token 下标对。
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from training.common import OP2IDX, SPECIAL_TOKENS, serialize_observation

MAX_LEN = 4096


class JevDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer, max_len: int = MAX_LEN):
        self.samples = samples
        self.tok = tokenizer
        self.max_len = max_len
        self.elem_tid = tokenizer.convert_tokens_to_ids("<|elem|>")
        self.decide_tid = tokenizer.convert_tokens_to_ids("<|decide|>")
        self.goal_tid = tokenizer.convert_tokens_to_ids("<|goal|>")
        self.end_goal_tid = tokenizer.convert_tokens_to_ids("<|end_goal|>")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        text = serialize_observation(s["goal"], s["foreground"], s["elements"], s["history"])
        enc = self.tok(text, return_offsets_mapping=True, truncation=True,
                       max_length=self.max_len)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        # 锚点位置
        elem_positions = [i for i, t in enumerate(ids) if t == self.elem_tid]
        decide_pos = ids.index(self.decide_tid) if self.decide_tid in ids else len(ids) - 1
        # 元素槽位可能被 4096 截断：截断后丢失的标签槽 → 指向 no-element 槽
        elem_slot = s["elem_slot"]
        if elem_slot < len(elem_positions):
            elem_label = elem_slot
        else:
            elem_label = len(elem_positions)  # no-element 槽

        # goal 区 token 范围（span 打分限定在 goal 内）
        goal_start = ids.index(self.goal_tid) if self.goal_tid in ids else 0
        goal_end = ids.index(self.end_goal_tid) if self.end_goal_tid in ids else goal_start

        # span 字符区间 → token 下标（相对 goal 起点的偏移在序列化文本中同样成立）
        span_label = [-1, -1]
        has_span = 0
        if s["span"] is not None:
            cs, ce = s["span"]
            # 序列化文本以 "<|goal|>" 开头，goal 文本从 len("<|goal|>") 开始
            base = len("<|goal|>")
            ts = _char_to_token(offsets, base + cs)
            te = _char_to_token(offsets, base + ce - 1)
            if ts is not None and te is not None and goal_start <= ts <= te <= goal_end:
                span_label = [ts, te]
                has_span = 1

        return {
            "input_ids": ids,
            "elem_positions": elem_positions,
            "decide_pos": decide_pos,
            "goal_range": [goal_start, goal_end],
            "op_label": OP2IDX[s["op"]],
            "elem_label": elem_label,
            "n_elem_anchors": len(elem_positions),
            "span_label": span_label,
            "has_span": has_span,
            "quality": s["quality"],
        }


def _char_to_token(offsets, char_pos: int) -> int | None:
    for i, (a, b) in enumerate(offsets):
        if a <= char_pos < b:
            return i
    return None


def collate(batch: list[dict], pad_id: int) -> dict:
    """变长 padding；elem 锚点位置 ragged →  padded tensor + mask。"""
    max_len = max(len(b["input_ids"]) for b in batch)
    max_elem = max(b["n_elem_anchors"] for b in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    elem_pos = torch.full((B, max_elem), 0, dtype=torch.long)
    elem_mask = torch.zeros((B, max_elem), dtype=torch.bool)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = torch.tensor(b["input_ids"])
        attn[i, :n] = 1
        m = b["n_elem_anchors"]
        if m:
            elem_pos[i, :m] = torch.tensor(b["elem_positions"])
            elem_mask[i, :m] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "elem_pos": elem_pos,
        "elem_mask": elem_mask,
        "decide_pos": torch.tensor([b["decide_pos"] for b in batch]),
        "goal_range": torch.tensor([b["goal_range"] for b in batch]),
        "op_label": torch.tensor([b["op_label"] for b in batch]),
        # no-element 标签列重映射：样本级 no-element 槽=自身 n_anchors，
        # 但 batched logits 里 no-element logit 固定在最后一列 max_elem（loss=inf 根因）
        "elem_label": torch.tensor([
            b["elem_label"] if b["elem_label"] < b["n_elem_anchors"] else max_elem
            for b in batch
        ]),
        "span_label": torch.tensor([b["span_label"] for b in batch]),
        "has_span": torch.tensor([b["has_span"] for b in batch]),
    }


def build_tokenizer(base: str):
    """加载 tokenizer 并注册锚点特殊 token（模型侧需同步 resize embeddings）。"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
