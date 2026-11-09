"""sdpa 为什么比 eager 慢：测 mask=None / mask=全1 / 不同 attn 实现。

Qwen3 的 QK-norm + head_dim=128 在 Blackwell 上可能让 sdpa 走不到 flash 内核；
而"全 1 的 attention_mask"又会让它绕过 fast path。这里逐项量清楚，
决定 serve.py 里到底该传什么。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def bench(fn, n=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        xs.append((time.perf_counter() - t0) * 1000)
    xs.sort()
    return round(xs[len(xs) // 2], 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "v3" / "jev-Qwen3-0.6B-e2"))
    ap.add_argument("--tokens", default="1653,788,381")
    args = ap.parse_args()
    ckpt = Path(args.ckpt)
    import sys
    sys.path.insert(0, str(ROOT))
    from training.common import serialize_observation
    from training.dataset import build_tokenizer
    from training.model import JevModel
    import glob
    import json

    tok = build_tokenizer(str(ckpt))

    def real_els(n_want):
        files = sorted(glob.glob(str(Path.home() / ".screen_use" / "traces_full" / "*.jsonl")))
        for p in reversed(files):
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    els = json.loads(line).get("elements") or []
                    if len(els) >= min(n_want, 20):
                        return els[:n_want]
        return []

    for impl in ("sdpa", "eager"):
        m = JevModel(str(ckpt), resize_special=False, grad_checkpointing=False).to("cuda")
        m.load_state_dict(torch.load(ckpt / "heads.pt", map_location="cuda",
                                     weights_only=True), strict=False)
        m.backbone.config._attn_implementation = impl
        m.eval()
        print(f"\n=== attn={impl} ===")
        for n_el in (84, 40, 20):
            slim = [{k: e.get(k) for k in ("id", "name", "control_type", "disabled",
                                           "checked", "expanded", "selected")}
                    for e in real_els(n_el)]
            text = serialize_observation("把窗口关掉", "记事本", slim, [])
            ids = tok(text, truncation=True, max_length=2048)["input_ids"]
            t = torch.tensor([ids], device="cuda")
            ones = torch.ones((1, len(ids)), device="cuda")
            with torch.inference_mode():
                r_ones = bench(lambda: m.backbone(input_ids=t, attention_mask=ones))
                try:
                    r_none = bench(lambda: m.backbone(input_ids=t, attention_mask=None))
                except Exception as e:  # noqa: BLE001
                    r_none = f"fail: {type(e).__name__}"
                try:
                    r_mask = bench(lambda: m.backbone(input_ids=t,
                                                     attention_mask=torch.ones_like(t, dtype=torch.bool)))
                except Exception as e:  # noqa: BLE001
                    r_mask = f"fail: {type(e).__name__}"
            print(f"  n_el={n_el:>3} tokens={len(ids):>5}  mask=float32_ones {r_ones:>8} ms"
                  f"   mask=None {r_none}   mask=bool_ones {r_mask}")
        del m
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
