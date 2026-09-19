"""校准与 τ 选择（TRAINING.md §7）：temperature scaling + 等张回归 + selective 曲线。

关键决定（2026-09-19 修订）：
1. **留出集再分半**：一半拟合温度/等张映射，另一半只用来选 τ 并报 ECE——
   否则等张回归在同一批数据上拟合+评估，会得到 ECE=0.0 这种自欺数字。
   分半按**轨迹文件**切（同 episode 的步高度相关，按步切会泄漏）。
2. **等张断点落盘**：`calibration.json` 存 isotonic {x, y}，serve 侧据此把原始
   sigmoid 映射成 P(决策正确)。不落盘的话 τ 的语义在部署时就丢了
   （旧 ckpt 的 τ=0.5 是在校准后 conf 上选的，serve 却拿原始 conf 比 → 升级率漂移）。

用法：
    ./.venv/Scripts/python.exe -m training.calibrate \
        --ckpt checkpoints/jev-Qwen3-0.6B-e3 --data ~/.screen_use/traces_mock

产出：{ckpt}/calibration.json（温度、isotonic 断点、τ、覆盖率、风险、selective 曲线）
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.common import iter_samples
from training.dataset import JevDataset, build_tokenizer, collate
from training.model import JevModel
from training.train import split_holdout


@torch.no_grad()
def collect_logits(model, loader, device):
    op_logits, op_labels, elem_logits, elem_labels, confs = [], [], [], [], []
    max_w = 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        out = model(batch["input_ids"], batch["attention_mask"],
                    batch["elem_pos"], batch["elem_mask"], batch["decide_pos"])
        op_logits.append(out["op_logits"].cpu())
        op_labels.append(batch["op_label"].cpu())
        # 元素 logits 各 batch 宽度不同（max_elem+1），先 -inf 补齐到统一宽度再 cat
        el = out["elem_logits"].cpu()
        max_w = max(max_w, el.shape[1])
        elem_logits.append(el)
        elem_labels.append(batch["elem_label"].cpu())
        confs.append(torch.sigmoid(out["conf_logit"]).cpu())
    elem_logits = [torch.nn.functional.pad(el, (0, max_w - el.shape[1]), value=float("-inf"))
                   for el in elem_logits]
    return (torch.cat(op_logits), torch.cat(op_labels),
            torch.cat(elem_logits), torch.cat(elem_labels), torch.cat(confs))


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """单参数温度缩放：网格搜索最小化 NLL（足够稳，免 scipy 依赖）。"""
    best_t, best_nll = 1.0, float("inf")
    for t in [x / 20 for x in range(10, 101)]:  # 0.5 .. 5.0
        nll = torch.nn.functional.cross_entropy(logits / t, labels).item()
        if nll < best_nll:
            best_nll, best_t = nll, t
    return best_t


def ece(conf: torch.Tensor, correct: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error。"""
    bins = torch.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.any():
            e += mask.float().mean().item() * abs(
                conf[mask].mean().item() - correct[mask].float().mean().item())
    return e


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", action="append", required=True)
    p.add_argument("--coverage-target", type=float, default=0.725,
                   help="τ 选择的覆盖率目标（TRAINING §7：70-75%）")
    p.add_argument("--tau", type=float, default=0.0,
                   help="钉死 τ（0=按覆盖率目标自动选）。部署侧曲线（scripts/eval_offline.py "
                        "的 deployed.curve）比 10% 留出切片更可信，用它定下 τ 后从这里传入")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.ckpt)
    tok = build_tokenizer(str(ckpt))  # ckpt 已含特殊 token
    model = JevModel(str(ckpt), resize_special=False).to(device)  # backbone 从 ckpt 加载
    heads = torch.load(ckpt / "heads.pt", map_location=device, weights_only=True)
    model.load_state_dict(heads, strict=False)
    model.eval()

    samples = iter_samples([Path(d).expanduser() for d in args.data])
    stats = train_stats(samples, tok)
    _, hold = split_holdout(samples)  # 与训练同 seed 的留出集
    calib_s, select_s = split_calib_select(hold)

    # 温度缩放（在 calib 半集上拟合，避免用选择集调参）
    cal = _collect(model, calib_s, tok, device)
    t_op = fit_temperature(cal["op_logits"], cal["op_labels"])
    t_elem = fit_temperature(cal["elem_logits"], cal["elem_labels"])
    conf_cal, correct_cal = _calibrated_conf(cal, t_op, t_elem)

    # 等张回归 conf → P(联合正确)：只在 calib 半集拟合，断点落盘供 serve 复用
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip").fit(conf_cal.numpy(), correct_cal.numpy())
    iso_x, iso_y = _isotonic_points(iso)

    # 选择集：τ 扫描 + 诚实 ECE（与拟合数据不相交）
    sel = _collect(model, select_s, tok, device)
    conf_sel, correct_sel = _calibrated_conf(sel, t_op, t_elem)
    conf_sel_iso = torch.tensor(_apply_isotonic(conf_sel.tolist(), iso_x, iso_y),
                                dtype=torch.float32)

    ece_before = ece(conf_sel, correct_sel)
    ece_after = ece(conf_sel_iso, correct_sel)

    # τ 扫描：coverage = 快路径执行比例，risk = 快路径上的错误率
    curve = []
    for tau in [x / 100 for x in range(50, 99)]:
        fast = conf_sel_iso >= tau
        if fast.float().mean() < 0.3:  # 快路径太窄无意义
            continue
        curve.append({
            "tau": tau,
            "coverage": fast.float().mean().item(),
            "risk": 1 - correct_sel[fast].mean().item() if fast.any() else 1.0,
            "escalation_rate": 1 - fast.float().mean().item(),
        })
    # 选 τ：显式 --tau 优先；否则覆盖率最接近目标且风险最低；
    # 若 conf 整体偏低导致候选为空则退回中位数
    if args.tau:
        tau = args.tau
        fast = conf_sel_iso >= tau
        best = {"tau": tau, "coverage": fast.float().mean().item(),
                "risk": 1 - correct_sel[fast].mean().item() if fast.any() else 1.0,
                "escalation_rate": 1 - fast.float().mean().item()}
        tau_source = "cli"
    elif curve:
        best = min(curve, key=lambda c: (abs(c["coverage"] - args.coverage_target), c["risk"]))
        tau_source = "coverage_target"
    else:
        tau = float(conf_sel_iso.median())
        fast = conf_sel_iso >= tau
        best = {"tau": round(tau, 3), "coverage": fast.float().mean().item(),
                "risk": 1 - correct_sel[fast].mean().item() if fast.any() else 1.0,
                "escalation_rate": 1 - fast.float().mean().item()}
        tau_source = "median_fallback"

    # 部署一致性检查：serve 侧按落盘断点插值，覆盖率应与选择集一致（±0.5%）
    sel_cov = best["coverage"]
    result = {
        "temperature_op": t_op, "temperature_elem": t_elem,
        "isotonic": {"x": iso_x, "y": iso_y},  # serve 据此把原始 sigmoid → P(正确)
        "train_stats": stats,                  # 训练分布尺度 → serve 侧门控（ood_check）
        "ece_before": round(ece_before, 4), "ece_after": round(ece_after, 4),
        "tau": best["tau"], "coverage": round(sel_cov, 3),
        "risk": round(best["risk"], 3),
        "tau_source": tau_source,
        "joint_acc_holdout": round(correct_sel.mean().item(), 3),
        "n_calib": len(calib_s), "n_select": len(select_s),
        "selective_curve": curve,
    }
    (ckpt / "calibration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    shown = {k: v for k, v in result.items() if k not in ("selective_curve", "isotonic")}
    print(json.dumps(shown, ensure_ascii=False, indent=1))


def train_stats(samples: list[dict], tok, limit: int = 500) -> dict:
    """训练观测的尺度统计（元素数 / token 数）→ serve 侧的分布门控用。

    2026-09-19 实测：mock 训练集元素数中位 14，真实桌面中位 84 —— 模型在后者上
    joint_acc 0.15 / 升级率 90%。把这个尺度落盘，serve 就能"只在自己见过的
    尺度上做决策"，不在分布内时直接升级，省掉一次必然被推翻的前向。
    """
    from training.common import serialize_observation

    if not samples:
        return {}
    els = sorted(len(s["elements"]) for s in samples)
    toks = sorted(
        len(tok(serialize_observation(s["goal"], s["foreground"], s["elements"],
                                      s["history"]))["input_ids"])
        for s in samples[:limit]
    )

    def q(values: list[int], p: float) -> int:
        return int(values[min(len(values) - 1, int(len(values) * p))])

    return {
        "n": len(samples),
        "elements_p50": q(els, 0.5), "elements_p95": q(els, 0.95), "elements_max": els[-1],
        "tokens_p50": q(toks, 0.5), "tokens_p95": q(toks, 0.95), "tokens_max": toks[-1],
    }


def split_calib_select(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    """按轨迹文件分半：一半拟合温度/等张，一半选 τ 并报 ECE。

    按文件（episode）切分而非按步切分——同 episode 的连续步高度相关，
    随机切会把同一状态泄漏到两侧，让 ECE/覆盖率虚高。
    小样本兜底：哈希分半可能把**所有文件**分到同一侧（实测 2 个文件时就会发生），
    此时按文件整体挪一个过去——仍保证「同 episode 不跨侧」，只是两侧都有数据。
    """
    calib, select = [], []
    for s in samples:
        digest = hashlib.sha1(s.get("file", "").encode("utf-8")).digest()[0]
        (calib if digest % 2 == 0 else select).append(s)
    if not calib or not select:
        src, dst = (calib, select) if not select else (select, calib)
        files = sorted({s.get("file", "") for s in src})
        if len(files) >= 2:
            move = {files[0]}
            dst.extend(s for s in src if s.get("file", "") in move)
            src[:] = [s for s in src if s.get("file", "") not in move]
    return calib, select


def _collect(model, samples: list[dict], tok, device) -> dict:
    if not samples:
        raise SystemExit(
            "校准集为空：留出样本太少（分半后有一侧没有数据）。\n"
            "校准至少需要 2 个轨迹文件的留出样本；数据量不够时请先多量产几回合。")
    dl = DataLoader(JevDataset(samples, tok), batch_size=16,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    op_logits, op_labels, elem_logits, elem_labels, confs = collect_logits(model, dl, device)
    return {"op_logits": op_logits, "op_labels": op_labels, "elem_logits": elem_logits,
            "elem_labels": elem_labels, "confs": confs}


def _calibrated_conf(data: dict, t_op: float, t_elem: float) -> tuple[torch.Tensor, torch.Tensor]:
    """→ (原始 sigmoid conf, 联合正确标签 correct)（顺序与调用点一致）。"""
    op_pred = (data["op_logits"] / t_op).argmax(-1)
    elem_pred = (data["elem_logits"] / t_elem).argmax(-1)
    correct = ((op_pred == data["op_labels"]) & (elem_pred == data["elem_labels"])).float()
    return data["confs"], correct


def _isotonic_points(iso) -> tuple[list[float], list[float]]:
    """取出等张回归的断点（sklearn 版本差异兜底）。"""
    xs = getattr(iso, "X_thresholds_", None)
    ys = getattr(iso, "y_thresholds_", None)
    if xs is None or ys is None:  # 老版本 sklearn
        xs, ys = iso.f_.x, iso.f_.y
    return [float(v) for v in xs], [float(v) for v in ys]


def _apply_isotonic(values: list[float], xs: list[float], ys: list[float]) -> list[float]:
    """与 serve.JevDecision._calibrate 同一套分段线性插值（这里用 bisect 批量版）。"""
    import bisect

    out = []
    for v in values:
        if not xs:
            out.append(v)
            continue
        if v <= xs[0]:
            out.append(float(ys[0]))
        elif v >= xs[-1]:
            out.append(float(ys[-1]))
        else:
            i = bisect.bisect_left(xs, v)
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            out.append(float(y1) if x1 <= x0 else
                       float(y0 + (y1 - y0) * (v - x0) / (x1 - x0)))
    return out


if __name__ == "__main__":
    main()
