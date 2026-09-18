"""serve：Jev 推理封装（观测进 → 决策出），供 VisualLoop/run_task 调用。

用法：
    from training.serve import JevDecision
    jev = JevDecision("checkpoints/jev-Qwen3-0.6B-e3")
    d = jev.decide(goal, foreground, elements, history)
    # → {"action": "CLICK", "element_id": 34, "confidence": 0.92, "escalate": False}
    # escalate=True 时宿主/慢层接管（cascade protocol §4）
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from training.common import OPS, serialize_observation
from training.dataset import build_tokenizer, MAX_LEN
from training.model import JevModel


class JevDecision:
    def __init__(self, ckpt: str | Path, device: str | None = None):
        ckpt = Path(ckpt)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = build_tokenizer(str(ckpt))
        self.model = JevModel(str(ckpt), resize_special=False,
                              grad_checkpointing=False).to(self.device)
        heads = torch.load(ckpt / "heads.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(heads, strict=False)
        self.model.eval()

        calib_path = ckpt / "calibration.json"
        self.calib = json.loads(calib_path.read_text()) if calib_path.exists() else {}
        self.tau = self.calib.get("tau", 0.9)  # 未校准时保守：大部分升级
        self.t_op = self.calib.get("temperature_op", 1.0)
        self.t_elem = self.calib.get("temperature_elem", 1.0)

        self.elem_tid = self.tok.convert_tokens_to_ids("<|elem|>")
        self.decide_tid = self.tok.convert_tokens_to_ids("<|decide|>")

    @torch.no_grad()
    def decide(self, goal: str, foreground: str, elements: list[dict],
               history: list[str]) -> dict:
        t0 = time.perf_counter()
        text = serialize_observation(goal, foreground, elements, history)
        ids = self.tok(text, truncation=True, max_length=MAX_LEN)["input_ids"]
        elem_positions = [i for i, t in enumerate(ids) if t == self.elem_tid]
        decide_pos = ids.index(self.decide_tid) if self.decide_tid in ids else len(ids) - 1
        n = len(elem_positions)

        input_ids = torch.tensor([ids], device=self.device)
        attn = torch.ones_like(input_ids)
        elem_pos = torch.tensor([elem_positions or [0]], device=self.device)
        elem_mask = torch.tensor([[n > 0] * max(n, 1)], device=self.device)
        decide = torch.tensor([decide_pos], device=self.device)

        out = self.model(input_ids, attn, elem_pos, elem_mask, decide)
        op_probs = torch.softmax(out["op_logits"][0] / self.t_op, -1)
        elem_probs = torch.softmax(out["elem_logits"][0] / self.t_elem, -1)
        op_idx = int(op_probs.argmax())
        elem_idx = int(elem_probs.argmax())
        conf = float(torch.sigmoid(out["conf_logit"][0]))

        action = OPS[op_idx]
        element_id = None
        if elem_idx < n:  # 命中具体元素（no-element 槽 = n）
            element_id = elements[elem_idx]["id"]
        elif action in ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
            # 需要元素却选了 no-element → 不自信，升级
            conf = min(conf, 0.3)

        return {
            "action": action,
            "element_id": element_id,
            "confidence": conf,
            "escalate": conf < self.tau or action == "BLOCKED",
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }
