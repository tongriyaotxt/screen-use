"""SFT 训练入口（TRAINING.md §5/§6）。

用法（项目根，.venv）：
    ./.venv/Scripts/python.exe -m training.train \
        --data ~/.screen_use/traces_mock --data ~/.screen_use/traces_full \
        --base Qwen/Qwen3-0.6B --epochs 3 --batch 8 --accum 4

产出：checkpoints/jev-{base}-e{epoch}/（模型+tokenizer）+ 每 epoch 留出集指标。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.common import iter_samples
from training.dataset import JevDataset, build_tokenizer, collate
from training.model import JevModel, compute_loss


def split_holdout(samples: list[dict], holdout_frac: float = 0.1, seed: int = 0):
    """按 episode 文件划分（同一任务的步不跨训练/留出，防泄漏）。"""
    files = sorted({s["file"] for s in samples})
    random.Random(seed).shuffle(files)
    n_hold = max(1, int(len(files) * holdout_frac))
    hold_files = set(files[:n_hold])
    train = [s for s in samples if s["file"] not in hold_files]
    hold = [s for s in samples if s["file"] in hold_files]
    return train, hold


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    n = op_ok = joint_ok = 0
    brier = 0.0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        out = model(batch["input_ids"], batch["attention_mask"],
                    batch["elem_pos"], batch["elem_mask"], batch["decide_pos"])
        op_pred = out["op_logits"].argmax(-1)
        elem_pred = out["elem_logits"].argmax(-1)
        op_hit = op_pred == batch["op_label"]
        joint_hit = op_hit & (elem_pred == batch["elem_label"])
        conf = torch.sigmoid(out["conf_logit"])
        brier += ((conf - joint_hit.float()) ** 2).sum().item()
        n += len(op_hit)
        op_ok += op_hit.sum().item()
        joint_ok += joint_hit.sum().item()
    model.train()
    return {"n": n, "op_acc": op_ok / max(n, 1),
            "joint_acc": joint_ok / max(n, 1), "brier": brier / max(n, 1)}


def _pause_while_user_active(idle_s: float) -> None:
    """人在用电脑就暂停训练（owner：可以跑，但别影响正常使用）。

    每个优化步检查一次；暂停期间的日志会在恢复时补一行，便于事后看训练是不是被让路了。
    """
    from screen_use.winidle import user_idle_seconds

    if user_idle_seconds() >= idle_s:
        return
    t0 = time.time()
    print(f"[idle-gate] 检测到用户在用电脑，训练暂停……", flush=True)
    while user_idle_seconds() < idle_s:
        time.sleep(10)
    print(f"[idle-gate] 用户离开 {idle_s:.0f}s，继续训练"
          f"（本次让路 {time.time() - t0:.0f}s）", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", action="append", required=True,
                   help="轨迹目录（可多次指定）")
    p.add_argument("--base", default="Qwen/Qwen3-0.6B")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--max-samples", type=int, default=0, help="调试用：限制样本数")
    p.add_argument("--idle-gate-s", type=float, default=0.0,
                   help="用户空闲满这么多秒才继续训练（人在用就暂停；0=不设闸门）。"
                        "owner 要求「可以跑但别影响我用电脑」时用它")
    p.add_argument("--throttle-ms", type=int, default=0,
                   help="每个优化步后 sleep 这么多毫秒（给 GPU 让路，默认 0）")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    samples = iter_samples([Path(d).expanduser() for d in args.data])
    if args.max_samples:
        samples = samples[: args.max_samples]
    train_samples, hold_samples = split_holdout(samples)
    print(f"样本: {len(samples)}（训练 {len(train_samples)} / 留出 {len(hold_samples)}）",
          flush=True)

    tok = build_tokenizer(args.base)
    model = JevModel(args.base).to(device)
    train_ds = JevDataset(train_samples, tok)
    hold_ds = JevDataset(hold_samples, tok)
    pad = tok.pad_token_id
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=lambda b: collate(b, pad))
    hold_dl = DataLoader(hold_ds, batch_size=args.batch,
                         collate_fn=lambda b: collate(b, pad))

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr)
        print("optimizer: AdamW8bit", flush=True)
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        print("optimizer: AdamW（bitsandbytes 不可用，退回普通版）", flush=True)
    total_steps = max(1, len(train_dl) * args.epochs // args.accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    model.train()
    step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"],
                        batch["elem_pos"], batch["elem_mask"], batch["decide_pos"])
            loss, metrics = compute_loss(out, batch)
            (loss / args.accum).backward()
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if args.idle_gate_s:
                    _pause_while_user_active(args.idle_gate_s)
                if args.throttle_ms:
                    time.sleep(args.throttle_ms / 1000)
                if step % 10 == 0:
                    print(f"e{epoch} s{step}/{total_steps} loss={loss.item():.3f} "
                          f"op={metrics['acc_op']:.2f} joint={metrics['acc_joint']:.2f}",
                          flush=True)
        metrics = evaluate(model, hold_dl, device)
        print(f"== epoch {epoch} ({time.time() - t0:.0f}s) 留出集: {metrics}", flush=True)
        ckpt = out_dir / f"jev-{args.base.split('/')[-1]}-e{epoch}"
        model.backbone.save_pretrained(ckpt)
        torch.save({k: v for k, v in model.state_dict().items()
                    if not k.startswith("backbone.")}, ckpt / "heads.pt")
        tok.save_pretrained(ckpt)
        (ckpt / "train_args.json").write_text(json.dumps(vars(args)), encoding="utf-8")
        print(f"saved → {ckpt}", flush=True)


if __name__ == "__main__":
    main()
