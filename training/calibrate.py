"""校准与 τ 选择（TRAINING.md §7）：temperature scaling + 等张回归 + selective 曲线。

用法：
    ./.venv/Scripts/python.exe -m training.calibrate \
        --ckpt checkpoints/jev-Qwen3-0.6B-e3 --data ~/.screen_use/traces_mock

产出：{ckpt}/calibration.json {temperature, tau, coverage_at_tau, risk_at_tau}
"""

from __future__ import annotations

import argparse
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
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.ckpt)
    base = json.loads((ckpt / "train_args.json").read_text())["base"]
    tok = build_tokenizer(str(ckpt))  # ckpt 已含特殊 token
    model = JevModel(str(ckpt), resize_special=False).to(device)  # backbone 从 ckpt 加载
    heads = torch.load(ckpt / "heads.pt", map_location=device, weights_only=True)
    model.load_state_dict(heads, strict=False)
    model.eval()

    samples = iter_samples([Path(d).expanduser() for d in args.data])
    _, hold = split_holdout(samples)  # 与训练同 seed 的留出集
    dl = DataLoader(JevDataset(hold, tok), batch_size=16,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    op_logits, op_labels, elem_logits, elem_labels, conf = collect_logits(model, dl, device)

    # 温度缩放（分别拟合 op / elem）
    t_op = fit_temperature(op_logits, op_labels)
    t_elem = fit_temperature(elem_logits, elem_labels)
    op_pred = (op_logits / t_op).argmax(-1)
    elem_pred = (elem_logits / t_elem).argmax(-1)
    correct = ((op_pred == op_labels) & (elem_pred == elem_labels)).float()

    # 置信度改用校准后联合正确的经验频率：等张回归（conf → P(correct)）
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip").fit(conf.numpy(), correct.numpy())
    conf_cal = torch.tensor(iso.predict(conf.numpy()), dtype=torch.float32)

    ece_before = ece(conf, correct)
    ece_after = ece(conf_cal, correct)

    # τ 扫描：coverage = 快路径执行比例，risk = 快路径上的错误率
    curve = []
    for tau in [x / 100 for x in range(50, 99)]:
        fast = conf_cal >= tau
        if fast.float().mean() < 0.3:  # 快路径太窄无意义
            continue
        curve.append({
            "tau": tau,
            "coverage": fast.float().mean().item(),
            "risk": 1 - correct[fast].mean().item() if fast.any() else 1.0,
            "escalation_rate": 1 - fast.float().mean().item(),
        })
    # 选 τ：覆盖率最接近目标且风险最低；若 conf 整体偏低导致候选为空则退回中位数
    if curve:
        best = min(curve, key=lambda c: (abs(c["coverage"] - args.coverage_target), c["risk"]))
    else:
        tau = float(conf_cal.median())
        fast = conf_cal >= tau
        best = {"tau": round(tau, 3), "coverage": fast.float().mean().item(),
                "risk": 1 - correct[fast].mean().item() if fast.any() else 1.0,
                "escalation_rate": 1 - fast.float().mean().item()}

    result = {
        "temperature_op": t_op, "temperature_elem": t_elem,
        "ece_before": round(ece_before, 4), "ece_after": round(ece_after, 4),
        "tau": best["tau"], "coverage": round(best["coverage"], 3),
        "risk": round(best["risk"], 3),
        "joint_acc_holdout": round(correct.mean().item(), 3),
        "selective_curve": curve,
    }
    (ckpt / "calibration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "selective_curve"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
