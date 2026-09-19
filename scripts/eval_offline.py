"""TRAINING §7 离线评测：多切片留出集 + 部署一致性检查。

为什么不是单次留出集：mock 回合是「模板同质」的（一个 episode 一个范式），
10% 文件切片只含 ~110 个回合，切片间 joint_acc 能差 10 个点以上——
只报一条切片等于报噪声。本脚本对多个切片求 mean±sd，并单独报真实轨迹子集。

同时做**部署一致性检查**（2026-09-19 修的那个坑）：
τ 是在等张校准后的 conf 上选的，因此 serve 侧必须用同一套映射；
这里用真实 `JevDecision.decide()` 跑一遍，对比「离线覆盖率 vs 线上升级率」。

用法：
    ./.venv/Scripts/python.exe scripts/eval_offline.py --ckpt checkpoints/jev-Qwen3-0.6B-e3 \
        --data ~/.screen_use/traces_mock --data ~/.screen_use/traces_full --seeds 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # training/ 未安装进 venv

from training.calibrate import _calibrated_conf, _collect, ece, fit_temperature  # noqa: E402
from training.common import iter_samples  # noqa: E402
from training.dataset import build_tokenizer  # noqa: E402
from training.model import JevModel  # noqa: E402
from training.train import split_holdout  # noqa: E402


def eval_slice(model, tok, device, samples: list[dict], seed: int) -> dict:
    _, hold = split_holdout(samples, seed=seed)
    if not hold:
        return {}
    t_op, t_elem = 0.0, 0.0
    d = _collect(model, hold, tok, device)
    t_op = fit_temperature(d["op_logits"], d["op_labels"])
    t_elem = fit_temperature(d["elem_logits"], d["elem_labels"])
    op_pred = (d["op_logits"] / t_op).argmax(-1)
    conf, correct = _calibrated_conf(d, t_op, t_elem)
    return {
        "seed": seed, "n": len(hold), "op_acc": (op_pred == d["op_labels"]).float().mean().item(),
        "joint_acc": correct.mean().item(),
        "ece_before": ece(conf, correct),
        "t_op": t_op, "t_elem": t_elem,
    }


def deployed_check(ckpt: str, samples: list[dict], limit: int) -> dict:
    """走真实 serve 路径：τ + 等张映射 + 单步延迟（这才是线上会发生的事）。

    抽样等距跨全集（不是取前 N）：不同来源/不同合成器版本的 conf 分布不同，
    取前缀会把「部署一致性」测成某个子集的偏差。
    """
    from training.serve import JevDecision

    jev = JevDecision(ckpt)
    stride = max(1, len(samples) // limit)
    picked = samples[::stride][:limit]
    escalations, lat, op_ok, joint_ok = 0, [], 0, 0
    per_source: dict[str, list[int]] = {}
    pairs: list[tuple[float, bool]] = []  # (校准后 conf, 联合正确) → 部署侧 selective 曲线
    for s in picked:
        t0 = time.perf_counter()
        d = jev.decide(s["goal"], s["foreground"], s["elements"], list(s["history"]))
        lat.append((time.perf_counter() - t0) * 1000)
        escalations += bool(d["escalate"])
        per_source.setdefault(s.get("source", "?"), []).append(int(not d["escalate"]))
        target_id = (s["elements"][s["elem_slot"]]["id"]
                     if s["elem_slot"] < len(s["elements"]) else None)
        op_hit = d["action"] == s["op"]
        joint_hit = bool(op_hit and d["element_id"] == target_id)
        pairs.append((float(d["confidence"]), joint_hit))
        op_ok += op_hit
        joint_ok += joint_hit
    n = max(len(picked), 1)
    lat_sorted = sorted(lat)
    return {
        "n": len(picked), "tau": jev.tau, "isotonic": bool(jev._iso_x),
        "escalation_rate": round(escalations / n, 4),
        "coverage": round(1 - escalations / n, 4),
        "coverage_by_source": {k: round(sum(v) / len(v), 4)
                               for k, v in sorted(per_source.items())},
        "op_acc": round(op_ok / n, 4), "joint_acc": round(joint_ok / n, 4),
        "latency_ms": {
            "mean": round(statistics.mean(lat), 1) if lat else 0.0,
            "p50": round(lat_sorted[len(lat_sorted) // 2], 1) if lat else 0.0,
            "p95": round(lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))], 1)
            if lat else 0.0,
        },
        "curve": _deployed_curve(pairs),
    }


def _deployed_curve(pairs: list[tuple[float, bool]], target: float = 0.725) -> dict:
    """部署侧 selective 曲线：全走真实 serve 的 conf，扫 τ 看覆盖率-风险。

    这个曲线才是选 τ 该看的东西：校准文件里的曲线只在 10% 留出切片上算，
    切片间覆盖率能差 10 个点以上（本文件 summary 里的 sd 就是证据）。
    """
    if not pairs:
        return {}
    curve = []
    for i in range(50, 96):
        tau = i / 100
        fast = [c for c, _ in pairs if c >= tau]
        if len(fast) < len(pairs) * 0.3:
            continue
        risk = 1 - sum(1 for c, ok in pairs if c >= tau and ok) / len(fast)
        curve.append({"tau": tau, "coverage": round(len(fast) / len(pairs), 4),
                      "risk": round(risk, 4),
                      "escalation_rate": round(1 - len(fast) / len(pairs), 4)})
    if not curve:
        return {}
    best = min(curve, key=lambda c: (abs(c["coverage"] - target), c["risk"]))
    return {"target_coverage": target, "recommended_tau": best["tau"],
            "recommended": best, "points": curve}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", action="append", required=True)
    p.add_argument("--seeds", type=int, default=5, help="留出集切片数（报 mean±sd）")
    p.add_argument("--latency-samples", type=int, default=50)
    p.add_argument("--out", default="")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = build_tokenizer(args.ckpt)
    model = JevModel(args.ckpt, resize_special=False).to(device)
    model.load_state_dict(
        torch.load(Path(args.ckpt) / "heads.pt", map_location=device, weights_only=True),
        strict=False)
    model.eval()

    samples = iter_samples([Path(d).expanduser() for d in args.data])
    slices = [s for s in (eval_slice(model, tok, device, samples, seed)
                          for seed in range(args.seeds)) if s]

    def stat(key: str) -> tuple[float, float]:
        vals = [s[key] for s in slices]
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    joint_m, joint_s = stat("joint_acc")
    op_m, op_s = stat("op_acc")
    ece_m, _ = stat("ece_before")
    result = {
        "ckpt": args.ckpt, "data": args.data, "samples": len(samples),
        "slices": [{"seed": s["seed"], "n": s["n"], "op_acc": round(s["op_acc"], 4),
                    "joint_acc": round(s["joint_acc"], 4),
                    "ece_before": round(s["ece_before"], 4),
                    "t_op": s["t_op"], "t_elem": s["t_elem"]} for s in slices],
        "summary": {
            "n_slices": len(slices),
            "op_acc": round(op_m, 4), "op_acc_sd": round(op_s, 4),
            "joint_acc": round(joint_m, 4), "joint_acc_sd": round(joint_s, 4),
            "ece_before": round(ece_m, 4),
        },
        "deployed": deployed_check(args.ckpt, samples, args.latency_samples),
    }
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                       encoding="utf-8")


if __name__ == "__main__":
    main()
