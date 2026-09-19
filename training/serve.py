"""serve：Jev 推理封装（观测进 → 决策出），供 VisualLoop/run_task 调用。

用法：
    from training.serve import JevDecision
    jev = JevDecision("checkpoints/jev-Qwen3-0.6B-e3")
    d = jev.decide(goal, foreground, elements, history)
    # → {"action": "CLICK", "element_id": 34, "text": None,
    #    "confidence": 0.92, "confidence_raw": 0.88, "escalate": False, "latency_ms": 41.2}
    # escalate=True 时宿主/慢层接管（cascade protocol §4）

单次前向、无解码：动作头 + 元素指针头 + span 头（TYPE 的 goal-copy）+ 置信度头。

置信度语义（2026-09-19 修正）：τ 是在**等张校准后**的 conf 上选的（calibrate.py），
因此这里也必须输出校准后的 conf，否则线上升级率与离线覆盖率对不上。
calibration.json 里没有 isotonic 断点（旧 ckpt）时退回原始 sigmoid，并在
返回值里用 confidence_raw 保留原值以便诊断。
"""

from __future__ import annotations

import bisect
import json
import time
from pathlib import Path

import torch

from training.common import OPS, serialize_observation
from training.dataset import build_tokenizer, MAX_LEN
from training.model import JevModel

_GOAL_TAG = "<|goal|>"
# 需要元素却选中 no-element 槽 → 明显不自信（压低置信交给升级协议）
_ELEMENT_OPS = ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK")
_LOW_CONF = 0.3


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
        iso = self.calib.get("isotonic") or {}
        self._iso_x: list[float] = list(iso.get("x", []))
        self._iso_y: list[float] = list(iso.get("y", []))

        self.elem_tid = self.tok.convert_tokens_to_ids("<|elem|>")
        self.decide_tid = self.tok.convert_tokens_to_ids("<|decide|>")
        self.goal_tid = self.tok.convert_tokens_to_ids(_GOAL_TAG)
        self.end_goal_tid = self.tok.convert_tokens_to_ids("<|end_goal|>")

    @torch.no_grad()
    def decide(self, goal: str, foreground: str, elements: list[dict],
               history: list[str]) -> dict:
        t0 = time.perf_counter()
        text = serialize_observation(goal, foreground, elements, history)
        enc = self.tok(text, truncation=True, max_length=MAX_LEN,
                       return_offsets_mapping=True)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
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
        conf_raw = float(torch.sigmoid(out["conf_logit"][0]))
        conf = self._calibrate(conf_raw)

        action = OPS[op_idx]
        element_id = None
        if elem_idx < n:  # 命中具体元素（no-element 槽 = n）
            element_id = elements[elem_idx]["id"]
        elif action in _ELEMENT_OPS:
            # 需要元素却选了 no-element → 不自信，升级
            conf = min(conf, _LOW_CONF)

        copied: str | None = None
        if action == "TYPE":
            copied = self._copy_text(goal, ids, offsets, out)
            if not copied:
                # span 抄不到文本 → 不许编文本（TRAINING §1），压低置信升级
                conf = min(conf, _LOW_CONF)

        return {
            "action": action,
            "element_id": element_id,
            "text": copied,
            "confidence": conf,
            "confidence_raw": conf_raw,
            "escalate": conf < self.tau or action == "BLOCKED",
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }

    # ---- 置信度校准（等张回归，训练侧 fit、推理侧只做插值） ----

    def _calibrate(self, conf_raw: float) -> float:
        """原始 sigmoid → P(决策正确)（calibration.json 的 isotonic 断点，线性插值）。"""
        xs, ys = self._iso_x, self._iso_y
        if not xs:
            return conf_raw  # 旧 ckpt：无可信映射，如实返回（不假装校准过）
        if conf_raw <= xs[0]:
            return float(ys[0])
        if conf_raw >= xs[-1]:
            return float(ys[-1])
        i = bisect.bisect_left(xs, conf_raw)
        x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
        if x1 <= x0:
            return float(y1)
        return float(y0 + (y1 - y0) * (conf_raw - x0) / (x1 - x0))

    # ---- TYPE 的 goal-copy（span 头 → goal 字符区间 → 文本） ----

    def _copy_text(self, goal: str, ids: list[int], offsets: list, out: dict) -> str | None:
        """span 头打分限定在 goal 区内（与训练 loss 同规则），命中区间即 copy 的文本。"""
        gs = ids.index(self.goal_tid) if self.goal_tid in ids else 0
        ge = ids.index(self.end_goal_tid) if self.end_goal_tid in ids else gs
        mask = torch.full_like(out["span_start_logits"][0], float("-inf"))
        mask[gs:ge + 1] = 0.0
        s_idx = int((out["span_start_logits"][0] + mask).argmax())
        e_idx = int((out["span_end_logits"][0] + mask).argmax())
        return span_to_text(goal, offsets, s_idx, max(s_idx, e_idx))


def span_to_text(goal: str, offsets: list, s_idx: int, e_idx: int) -> str | None:
    """token 下标对 → goal 内的字符文本（越界/空区间/落在特殊 token 上 → None）。"""
    base = len(_GOAL_TAG)
    try:
        a = offsets[s_idx][0] - base
        b = offsets[e_idx][1] - base
    except (IndexError, TypeError):
        return None
    if a < 0 or b > len(goal) or b <= a:
        return None
    return goal[a:b]
