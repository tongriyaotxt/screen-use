"""Jev 前向的微优化对照（只读，不改模型文件）。

同一张真实元素表，扫这些变量：
  baseline          现网：no_grad + 默认 attn 实现
  inference_mode    torch.inference_mode()（比 no_grad 少 autograd 簿记）
  attn=<impl>       显式指定 sdpa / flash_attention_2 / eager
  heads_bf16        六个线性头改成 bf16（省 hidden.float() 的转换与 fp32 GEMM）
  shortlist         只把候选压到 20/40 个元素（表长 → token 数）
输出 p50/p95，直接看哪条值得改。

用法：
    ./.venv/Scripts/python.exe scripts/probe_model_opts.py
    ./.venv/Scripts/python.exe scripts/probe_model_opts.py --ckpt checkpoints/v3/jev-Qwen3-0.6B-e2
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
MAX_LEN = 2048


def _pct(xs, q):
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


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
    return {"p50": round(_pct(xs, 0.5), 1), "p95": round(_pct(xs, 0.95), 1)}


def real_elements(n_want: int) -> list[dict]:
    files = sorted(glob.glob(str(Path.home() / ".screen_use" / "traces_full" / "*.jsonl")))
    for path in reversed(files):
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    els = json.loads(line).get("elements") or []
                    if len(els) >= min(n_want, 20):
                        return els[:n_want]
        except Exception:
            continue
    return [{"id": i + 1, "name": f"Button{i}", "control_type": "ButtonControl"}
            for i in range(n_want)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "v3" / "jev-Qwen3-0.6B-e2"))
    args = ap.parse_args()
    ckpt = Path(args.ckpt)

    import sys
    sys.path.insert(0, str(ROOT))
    from training.common import serialize_observation
    from training.dataset import build_tokenizer
    from training.model import JevModel

    tok = build_tokenizer(str(ckpt))
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"{torch.cuda.get_device_name(0)}")
    try:
        import flash_attn  # noqa: F401
        has_flash = True
    except Exception:
        has_flash = False
    print(f"flash_attn available: {has_flash}")

    base = JevModel(str(ckpt), resize_special=False, grad_checkpointing=False).to("cuda")
    base.load_state_dict(torch.load(ckpt / "heads.pt", map_location="cuda", weights_only=True),
                         strict=False)
    base.eval()
    print("attn impl (loaded):", getattr(base.backbone.config, "_attn_implementation", "?"))

    # 复刻 serve.decide 的张量准备
    def prep(els, goal="把窗口关掉", fg="记事本"):
        slim = [{k: e.get(k) for k in ("id", "name", "control_type", "disabled",
                                       "checked", "expanded", "selected")} for e in els]
        text = serialize_observation(goal, fg, slim, [])
        enc = tok(text, truncation=True, max_length=MAX_LEN)
        ids = enc["input_ids"]
        elem_tid = tok.convert_tokens_to_ids("<|elem|>")
        decide_tid = tok.convert_tokens_to_ids("<|decide|>")
        elem_positions = [i for i, t in enumerate(ids) if t == elem_tid]
        decide_pos = ids.index(decide_tid) if decide_tid in ids else len(ids) - 1
        n = len(elem_positions)
        return (torch.tensor([ids], device="cuda"),
                torch.ones((1, len(ids)), device="cuda"),
                torch.tensor([elem_positions or [0]], device="cuda"),
                torch.tensor([[n > 0] * max(n, 1)], device="cuda"),
                torch.tensor([decide_pos], device="cuda"),
                len(ids), n)

    results = {}

    for n_el in (84, 40, 20):
        els = real_elements(n_el)
        input_ids, attn, elem_pos, elem_mask, decide, n_tok, n_elem = prep(els)
        rows = {}

        def call():
            return base(input_ids, attn, elem_pos, elem_mask, decide)

        with torch.no_grad():
            rows["baseline_no_grad"] = bench(call)
        with torch.inference_mode():
            rows["inference_mode"] = bench(call)

        # 只量 backbone（头部的成本单独看）
        with torch.inference_mode():
            rows["backbone_only"] = bench(lambda: base.backbone(
                input_ids=input_ids, attention_mask=attn).last_hidden_state)

        # 头部成本：fp32 vs bf16
        with torch.inference_mode():
            hidden = base.backbone(input_ids=input_ids, attention_mask=attn).last_hidden_state

            def heads_fp32():
                h = hidden.float()
                arange = torch.arange(1, device=h.device)
                h_decide = h[arange, decide]
                base.op_head(h_decide)
                h_elem = h[arange.unsqueeze(1), elem_pos]
                base.elem_score(h_elem).squeeze(-1)
                base.span_start(h).squeeze(-1)
                base.span_end(h).squeeze(-1)
                base.conf_head(h_decide).squeeze(-1)

            rows["heads_fp32"] = bench(heads_fp32)
            h_bf = hidden.to(torch.bfloat16)
            heads = (base.op_head, base.elem_score, base.noelem_score,
                     base.span_start, base.span_end, base.conf_head)
            saved = [(p, p.weight.detach().clone(), p.bias.detach().clone() if p.bias is not None else None)
                     for p in heads]
            for p in heads:
                p.to(torch.bfloat16)

            def heads_bf16():
                arange = torch.arange(1, device=h_bf.device)
                h_decide = h_bf[arange, decide]
                base.op_head(h_decide)
                h_elem = h_bf[arange.unsqueeze(1), elem_pos]
                base.elem_score(h_elem).squeeze(-1)
                base.span_start(h_bf).squeeze(-1)
                base.span_end(h_bf).squeeze(-1)
                base.conf_head(h_decide).squeeze(-1)

            rows["heads_bf16"] = bench(heads_bf16)
            for p, w, b in saved:  # 还原
                p.to(torch.float32)
                with torch.no_grad():
                    p.weight.copy_(w)
                    if b is not None and p.bias is not None:
                        p.bias.copy_(b)

        results[f"n_elem={n_elem}"] = {"tokens": n_tok, "rows": rows}
        print(f"\n== {n_elem} elements / {n_tok} tokens ==")
        for k, v in rows.items():
            print(f"  {k:<20} p50 {v['p50']:>8.1f} ms   p95 {v['p95']:>8.1f} ms")

    # 注意力实现对照（重建模型，避免 config 混杂）
    for impl in ("sdpa", "eager", "flash_attention_2"):
        if impl == "flash_attention_2" and not has_flash:
            print(f"\n[skip] attn={impl}: 未安装 flash-attn")
            continue
        try:
            m = JevModel(str(ckpt), resize_special=False, grad_checkpointing=False).to("cuda")
            m.backbone.config._attn_implementation = impl
            m.load_state_dict(torch.load(ckpt / "heads.pt", map_location="cuda",
                                         weights_only=True), strict=False)
            m.eval()
            els = real_elements(84)
            input_ids, attn, elem_pos, elem_mask, decide, n_tok, _ = prep(els)
            with torch.inference_mode():
                r = bench(lambda: m(input_ids, attn, elem_pos, elem_mask, decide))
            print(f"\n[attn={impl}] {n_tok} tokens: p50 {r['p50']} ms  p95 {r['p95']} ms")
            results[f"attn={impl}"] = r
            del m
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"\n[attn={impl}] 失败: {e}")

    out = ROOT / "assets" / "bench" / "model_opts_20261109.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
