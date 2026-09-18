"""JevModel：Qwen3 backbone + 四个并行输出头（TRAINING.md §3）。

单次前向，无解码循环：
- 操作头：softmax 8 类（h_decide）
- 元素头：pointer-softmax over N 个 <|elem|> 锚点 + 1 个 no-element 槽（h_decide）
- 槽位头：goal 区 token 的 start/end 指针（copy 机制，非生成）
- 置信度头：标量 σ(h_decide)，Brier 联训（标签在线自举）
"""

from __future__ import annotations

import torch
import torch.nn as nn

from training.common import OPS, SPECIAL_TOKENS


class JevModel(nn.Module):
    def __init__(self, base: str = "Qwen/Qwen3-0.6B", n_ops: int = len(OPS),
                 grad_checkpointing: bool = True, resize_special: bool = True):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(base, dtype=torch.bfloat16)
        if resize_special:
            # 从基座初始化时扩词表；从 checkpoint 加载时（已扩过）跳过
            self.backbone.resize_token_embeddings(
                self.backbone.get_input_embeddings().weight.shape[0] + len(SPECIAL_TOKENS)
            )
        if grad_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        h = self.backbone.config.hidden_size
        self.op_head = nn.Linear(h, n_ops)
        self.elem_score = nn.Linear(h, 1)      # 每个 <|elem|> 锚点打分
        self.noelem_score = nn.Linear(h, 1)    # no-element 槽（h_decide）
        self.span_start = nn.Linear(h, 1)
        self.span_end = nn.Linear(h, 1)
        self.conf_head = nn.Linear(h, 1)
        # 头用 fp32 训练更稳（backbone bf16）
        for head in (self.op_head, self.elem_score, self.noelem_score,
                     self.span_start, self.span_end, self.conf_head):
            head.float()

    def forward(self, input_ids, attention_mask, elem_pos, elem_mask, decide_pos):
        hidden = self.backbone(input_ids=input_ids,
                               attention_mask=attention_mask).last_hidden_state
        hidden = hidden.float()
        B, L, H = hidden.shape
        arange = torch.arange(B, device=hidden.device)
        h_decide = hidden[arange, decide_pos]                    # [B, H]

        op_logits = self.op_head(h_decide)                       # [B, n_ops]

        # 元素头：收集锚点隐状态 → 打分；mask 掉的槽置 -inf；末尾拼 no-element 槽
        elem_pos = elem_pos.clamp(max=L - 1)
        h_elem = hidden[arange.unsqueeze(1), elem_pos]           # [B, max_elem, H]
        elem_logits = self.elem_score(h_elem).squeeze(-1)        # [B, max_elem]
        elem_logits = elem_logits.masked_fill(~elem_mask, float("-inf"))
        noelem_logits = self.noelem_score(h_decide)              # [B, 1]
        elem_logits = torch.cat([elem_logits, noelem_logits], dim=1)  # [B, max_elem+1]

        span_start_logits = self.span_start(hidden).squeeze(-1)  # [B, L]
        span_end_logits = self.span_end(hidden).squeeze(-1)      # [B, L]
        conf_logit = self.conf_head(h_decide).squeeze(-1)        # [B]

        return {
            "op_logits": op_logits,
            "elem_logits": elem_logits,
            "span_start_logits": span_start_logits,
            "span_end_logits": span_end_logits,
            "conf_logit": conf_logit,
        }


def compute_loss(out: dict, batch: dict, w_span: float = 0.5, w_conf: float = 0.5):
    """L = L_op + L_elem + 0.5·L_span + 0.5·L_conf（TRAINING §4）。

    conf 标签在线自举：y = 1{op 和 elem 均预测对}（detach，不回传）。
    span 打分限定在 goal 区内（区外置 -inf）。
    """
    op_label = batch["op_label"]
    elem_label = batch["elem_label"]
    l_op = nn.functional.cross_entropy(out["op_logits"], op_label)
    l_elem = nn.functional.cross_entropy(out["elem_logits"], elem_label)

    # span：仅 has_span 样本；goal 区外 -inf
    gs = batch["goal_range"][:, 0].unsqueeze(1)      # [B,1]
    ge = batch["goal_range"][:, 1].unsqueeze(1)
    positions = torch.arange(out["span_start_logits"].shape[1],
                             device=gs.device).unsqueeze(0)
    in_goal = (positions >= gs) & (positions <= ge)  # [B, L]
    neg_inf = float("-inf")
    ss = out["span_start_logits"].masked_fill(~in_goal, neg_inf)
    se = out["span_end_logits"].masked_fill(~in_goal, neg_inf)
    has_span = batch["has_span"].bool()
    if has_span.any():
        l_span = (nn.functional.cross_entropy(ss[has_span], batch["span_label"][has_span][:, 0])
                  + nn.functional.cross_entropy(se[has_span], batch["span_label"][has_span][:, 1]))
    else:
        l_span = torch.tensor(0.0, device=gs.device)

    with torch.no_grad():
        op_pred = out["op_logits"].argmax(-1)
        elem_pred = out["elem_logits"].argmax(-1)
        y_conf = ((op_pred == op_label) & (elem_pred == elem_label)).float()
    conf = torch.sigmoid(out["conf_logit"])
    l_conf = nn.functional.mse_loss(conf, y_conf)  # Brier

    loss = l_op + l_elem + w_span * l_span + w_conf * l_conf
    return loss, {"l_op": l_op.item(), "l_elem": l_elem.item(),
                  "l_span": float(l_span), "l_conf": l_conf.item(),
                  "acc_op": (op_pred == op_label).float().mean().item(),
                  "acc_joint": y_conf.mean().item()}
