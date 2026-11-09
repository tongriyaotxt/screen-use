"""延迟预算 + 硬件屋顶线探针（只读：不点鼠标、不改任何状态）。

回答一个问题：**每步到底把时间花在哪，理论地板在哪。**

分四块量：
  A 硬件天花板      —— 实测 bf16 TFLOPS / 显存带宽（决定"模型侧最快能多快"）
  B 感知            —— 截图 / 缩放+JPEG / UIA 冷枚举 / UIA 缓存命中 / 状态位读取
  C 决策            —— 真轨迹元素表 → tokenize / 前向 / 端到端 decide，按表长扫
  D 执行            —— SendInput 注入开销（dx=dy=0 的空移动，不移动光标）

用法：
    ./.venv/Scripts/python.exe scripts/bench_latency_budget.py
    ./.venv/Scripts/python.exe scripts/bench_latency_budget.py --ckpt checkpoints/v3/jev-Qwen3-0.6B-e2
    ./.venv/Scripts/python.exe scripts/bench_latency_budget.py --skip-uia --skip-exec
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pct(xs: list[float], q: float) -> float:
    """线性插值分位数（xs 未排序也可）。"""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def bench(fn, n: int = 20, warmup: int = 5, cuda: bool = False) -> dict:
    """测一段调用的毫秒分布（CUDA 时前后强制同步，否则只量到 kernel 派发）。"""
    for _ in range(warmup):
        fn()
    if cuda:
        import torch

        torch.cuda.synchronize()
    xs: list[float] = []
    for _ in range(n):
        if cuda:
            import torch

            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            import torch

            torch.cuda.synchronize()
        xs.append((time.perf_counter() - t0) * 1000)
    return {"n": n, "mean": round(statistics.fmean(xs), 2),
            "p50": round(_pct(xs, 0.5), 2), "p95": round(_pct(xs, 0.95), 2),
            "max": round(max(xs), 2)}


def fmt(name: str, r: dict, note: str = "") -> str:
    return (f"  {name:<34} p50 {r['p50']:>9.2f} ms   p95 {r['p95']:>9.2f} ms"
            f"   max {r['max']:>9.2f} ms   {note}")


# ============================ A 硬件天花板 ============================

def probe_hardware() -> dict:
    import torch

    out: dict = {"torch": torch.__version__, "cuda": torch.version.cuda}
    if not torch.cuda.is_available():
        out["device"] = "cpu"
        return out
    props = torch.cuda.get_device_properties(0)
    out.update({
        "device": props.name,
        "vram_gb": round(props.total_memory / 1024 ** 3, 1),
        "sm": props.multi_processor_count,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    })

    # bf16 matmul 实测 TFLOPS（4096³ = 137 GFLOP，足够压住噪声）
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)

    def mm():
        torch.mm(a, b)

    r = bench(mm, n=20, warmup=5, cuda=True)
    flops = 2 * 4096 ** 3
    out["bf16_tflops"] = round(flops / (r["p50"] / 1000) / 1e12, 1)
    out["bf16_matmul_ms"] = r["p50"]

    # fp16 对照
    a16 = a.half()
    b16 = b.half()

    def mm16():
        torch.mm(a16, b16)

    r16 = bench(mm16, n=20, warmup=5, cuda=True)
    out["fp16_tflops"] = round(flops / (r16["p50"] / 1000) / 1e12, 1)

    # 显存带宽：1 GiB 拷贝（读+写 2 GiB 流量）
    src = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)  # 256 MiB
    dst = torch.empty_like(src)

    def cp():
        dst.copy_(src)

    rc = bench(cp, n=20, warmup=5, cuda=True)
    gib = 256 / 1024
    out["hbm_gbps"] = round(2 * gib / (rc["p50"] / 1000), 0)
    return out


# ============================ B 感知 ============================

def probe_perception(skip_uia: bool) -> dict:
    out: dict = {}
    from screen_use.perception import screen

    out["screenshot"] = bench(lambda: screen.screenshot(), n=10, warmup=2)

    img = screen.screenshot()
    out["downscale_1280"] = bench(lambda: screen.downscale(img, 1280), n=10, warmup=2)
    small, _ = screen.downscale(img, 1280)
    out["jpeg_b64_80"] = bench(lambda: screen.to_jpeg_base64(small, 80), n=10, warmup=2)
    out["capture_for_model"] = bench(lambda: screen.capture_for_model(1280, 80), n=10, warmup=2)
    out["screen_size"] = list(img.size)

    if skip_uia:
        return out

    from screen_use.perception import uia_tree as ut

    ut.invalidate_window_cache()

    def cold():
        ut.invalidate_window_cache()
        return ut.list_ui_elements()

    els = cold()
    out["uia_cold"] = bench(cold, n=5, warmup=1)
    out["uia_warm_cached"] = bench(lambda: ut.list_ui_elements(), n=10, warmup=2)
    out["uia_force_full"] = bench(lambda: ut.list_ui_elements(force_full=True), n=5, warmup=1)
    out["elements"] = len(els)
    out["uia_timings"] = els.timings
    out["state_ms"] = els.timings.get("state_ms")

    # 状态位读取单独量：把已枚举的元素重新 annotate 一遍
    out["annotate_state"] = bench(lambda: ut._annotate_state(list(els)), n=5, warmup=1)
    return out


# ============================ C 决策 ============================

def _real_elements(n_want: int) -> list[dict]:
    """从真实轨迹里取一份元素表（优先最新的），按需截断到 n_want。"""
    import glob

    files = sorted(glob.glob(str(Path.home() / ".screen_use" / "traces_full" / "*.jsonl")))
    for path in reversed(files):
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    els = rec.get("elements") or []
                    if len(els) >= min(n_want, 20):
                        return els[:n_want]
        except Exception:
            continue
    # 兜底：合成表（宁可数字偏乐观也要能跑）
    return [{"id": i + 1, "name": f"Button{i}", "control_type": "ButtonControl",
             "bbox": [0, 0, 10, 10], "disabled": False, "checked": False,
             "expanded": False, "selected": False} for i in range(n_want)]


def probe_decision(ckpt: Path | None, sizes: list[int]) -> dict:
    if ckpt is None or not Path(ckpt).exists():
        return {"skipped": f"no ckpt at {ckpt}"}
    import torch

    sys.path.insert(0, str(ROOT))
    from training.serve import JevDecision

    t0 = time.perf_counter()
    jev = JevDecision(ckpt)
    load_s = time.perf_counter() - t0

    out: dict = {"ckpt": str(ckpt), "load_s": round(load_s, 1),
                 "device": jev.device, "tau": jev.tau, "by_size": []}

    for n in sizes:
        els = _real_elements(n)
        # 元素表里只留决策需要的字段（真实 decide 就是吃这个）
        slim = [{k: e.get(k) for k in ("id", "name", "control_type", "disabled",
                                       "checked", "expanded", "selected")} for e in els]
        goal = "把窗口关掉"
        fg = "记事本"
        hist = ["1C #3", "2T #7 \"abc\""]

        d = jev.decide(goal, fg, slim, hist)  # warmup（含 CUDA 图/分配器预热）
        if d.get("action") == "OOD":
            out["by_size"].append({"elements": len(slim), "ood": True, "detail": d.get("detail")})
            continue

        from training.common import serialize_observation
        text = serialize_observation(goal, fg, slim, hist)
        n_tok = len(jev.tok(text)["input_ids"])

        r_decide = bench(lambda: jev.decide(goal, fg, slim, hist), n=10, warmup=3, cuda=True)
        r_tok = bench(lambda: jev.tok(text, truncation=True, max_length=4096), n=10, warmup=3)
        out["by_size"].append({"elements": len(slim), "tokens": n_tok,
                               "decide": r_decide, "tokenize": r_tok})

    # 纯前向 vs 精确度：bf16 backbone 已在 ckpt 里，这里量 fp32 对照的一次前向
    try:
        els = _real_elements(max(sizes))
        slim = [{k: e.get(k) for k in ("id", "name", "control_type")} for e in els]
        from training.common import serialize_observation
        text = serialize_observation("把窗口关掉", "记事本", slim, [])
        enc = jev.tok(text, truncation=True, max_length=4096, return_offsets_mapping=True)
        ids = enc["input_ids"]
        import torch as T

        input_ids = T.tensor([ids], device=jev.device)
        attn = T.ones_like(input_ids)
        elem_pos = T.tensor([[i for i, t in enumerate(ids) if t == jev.elem_tid] or [0]],
                            device=jev.device)
        elem_mask = T.tensor([[True]], device=jev.device)
        decide = T.tensor([len(ids) - 1], device=jev.device)
        r_fwd = bench(lambda: jev.model(input_ids, attn, elem_pos, elem_mask, decide),
                      n=10, warmup=3, cuda=True)
        out["backbone_only"] = {"tokens": len(ids), **r_fwd}

        # 同规模纯 matmul 的 FLOP 对照：模型每 token 约 2*0.6e9 FLOP
        out["model_flops_per_token"] = 1.2e9
    except Exception as e:  # noqa: BLE001 - 诊断脚本，量不到不影响主结论
        out["backbone_only_error"] = str(e)
    return out


# ============================ D 执行 ============================

def probe_exec() -> dict:
    """SendInput 注入开销（空移动 dx=dy=0：不发光标位移，只量系统调用 + 输入队列）。"""
    user32 = ctypes.windll.user32

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                    ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("mi", MOUSEINPUT)]

    inp = INPUT(type=0, mi=MOUSEINPUT(dx=0, dy=0, mouseData=0,
                                     dwFlags=0x0001,  # MOUSEEVENTF_MOVE
                                     time=0, dwExtraInfo=None))
    size = ctypes.sizeof(INPUT)

    def send_one():
        user32.SendInput(1, ctypes.byref(inp), size)

    out = {"sendinput_move": bench(send_one, n=50, warmup=5)}

    # pyautogui 的 PAUSE（每个调用固定 sleep）+ 一次 click 的固定睡眠成本
    import pyautogui

    out["pyautogui_pause_s"] = pyautogui.PAUSE
    out["click_fixed_ms"] = round(pyautogui.PAUSE * 1000 * 3, 1)  # move+down+up 各一次
    from screen_use.actions.executor import DEFAULT_RIGHT_HOLD_MS

    out["right_click_fixed_ms"] = round(DEFAULT_RIGHT_HOLD_MS + 0.08 * 1000 + 0.12 * 1000
                                        + pyautogui.PAUSE * 1000 * 3, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "v3" / "jev-Qwen3-0.6B-e2"))
    ap.add_argument("--sizes", default="14,40,84,150")
    ap.add_argument("--skip-uia", action="store_true")
    ap.add_argument("--skip-exec", action="store_true")
    ap.add_argument("--skip-decision", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    report: dict = {"ts": time.strftime("%Y-%m-%d %H:%M:%S")}

    print("[A] hardware ceiling")
    report["hardware"] = probe_hardware()
    for k, v in report["hardware"].items():
        print(f"  {k}: {v}")

    print("[B] perception (ms)")
    report["perception"] = probe_perception(args.skip_uia)
    for k, v in report["perception"].items():
        if isinstance(v, dict) and "p50" in v:
            print(fmt(k, v))
        elif k == "uia_timings":
            print(f"  uia_timings.total_ms = {v.get('total_ms')}, nodes = {v.get('nodes')},"
                  f" reused = {v.get('reused')}")
        elif k not in ("state_ms",):
            print(f"  {k}: {v}")

    print("[C] decision (ms)")
    report["decision"] = probe_decision(Path(args.ckpt) if args.ckpt else None, sizes)
    print(f"  load_s: {report['decision'].get('load_s')}  device: {report['decision'].get('device')}")
    for row in report["decision"].get("by_size", []):
        if row.get("ood"):
            print(f"  n={row['elements']:>4}  OOD gated ({row.get('detail')})")
            continue
        print(f"  n={row['elements']:>4} tokens={row['tokens']:>5} "
              f"decide p50 {row['decide']['p50']:>8.2f} p95 {row['decide']['p95']:>8.2f} | "
              f"tokenize p50 {row['tokenize']['p50']:>6.2f}")
    if "backbone_only" in report["decision"]:
        b = report["decision"]["backbone_only"]
        print(f"  backbone only: tokens={b['tokens']} p50 {b['p50']} p95 {b['p95']}")

    if not args.skip_exec:
        print("[D] execution (ms)")
        report["exec"] = probe_exec()
        for k, v in report["exec"].items():
            if isinstance(v, dict):
                print(fmt(k, v))
            else:
                print(f"  {k}: {v}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
