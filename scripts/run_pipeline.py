"""一条命令跑完数据→训练→评测：owner 离开电脑时用（要开窗 + 占 GPU）。

背景（`research/desktop-jev/BENCH.md` §7）：真实域上 Jev 的 joint_acc 只有 0.17、升级率 90%，
主因是训练观测的**尺度**与**语义**跟真实桌面差太远（元素中位 14 vs 84、goal 照抄控件名）。
mock v3 生成器就是补这两个差距的；本脚本把「量产 v3 → 重训 → 校准 → 真实域复测」串起来，
跑完直接给出对照表，不用人盯。

用法（项目根，.venv）：
    # 完整跑（默认 3000 个 v3 回合 / 3 epoch）
    ./.venv/Scripts/python.exe scripts/run_pipeline.py --i-am-away
    # 先小规模验证管线（200 回合 / 1 epoch，约 10 分钟）
    ./.venv/Scripts/python.exe scripts/run_pipeline.py --i-am-away --episodes 200 --epochs 1
    # 只重训+评测（v3 数据已经量产过）
    ./.venv/Scripts/python.exe scripts/run_pipeline.py --i-am-away --skip-gen

输出：assets/bench/logs/pipeline_<时间戳>.log + assets/bench/pipeline_<时间戳>.md（对照表）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
HOME = Path(os.environ.get("SCREEN_USE_HOME", Path.home() / ".screen_use"))


def run_step(name: str, argv: list[str], log, dry: bool = False) -> int:
    """跑一步（子进程），输出同时进日志与 stdout。返回退出码。"""
    head = f"\n{'=' * 70}\n[{time.strftime('%H:%M:%S')}] {name}\n{'=' * 70}"
    print(head, flush=True)
    log.write(head + "\n")
    log.write("$ " + " ".join(argv) + "\n")
    log.flush()
    if dry:
        print("(dry-run：跳过)", flush=True)
        return 0
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    # 让路优先：子进程降为 BelowNormal（CPU 侧不跟用户的应用抢），并抑制控制台闪窗
    flags = 0
    if os.name == "nt":
        flags = (getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
                 | getattr(subprocess, "CREATE_NO_WINDOW", 0))
    t0 = time.time()
    proc = subprocess.Popen(argv, cwd=str(ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            creationflags=flags)
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
    proc.wait()
    cost = time.time() - t0
    tail = f"[{name}] 退出码 {proc.returncode}，耗时 {cost / 60:.1f} 分钟"
    print(tail, flush=True)
    log.write(tail + "\n")
    log.flush()
    return proc.returncode


def count_episodes(d: Path) -> int:
    return len(list(d.glob("*.jsonl"))) if d.exists() else 0


def latest_ckpt(base: Path, epochs: int) -> Path | None:
    candidates = sorted(base.glob(f"*-e{epochs}"))
    return candidates[-1] if candidates else None


def estimate_training(data_dirs: list[Path], tokenizer_base: str, epochs: int) -> str:
    """粗估训练时长：按 token 量线性外推（v0 实测 3100 样本 × 525 token = 531 s/epoch）。

    只为了让人安排"离开多久"，不追求准；真实耗时以训练日志里的每 epoch 打印为准。
    """
    try:
        sys.path.insert(0, str(ROOT))
        from training.common import iter_samples, serialize_observation
        from training.dataset import build_tokenizer

        tok = build_tokenizer(tokenizer_base)
        total_samples, total_tokens, per_sample = 0, 0, []
        for d in data_dirs:
            ss = iter_samples([d])
            total_samples += len(ss)
            for s in ss[:200]:
                per_sample.append(len(tok(serialize_observation(
                    s["goal"], s["foreground"], s["elements"], s["history"]))["input_ids"]))
            if per_sample:
                total_tokens += (sum(per_sample) / len(per_sample)) * len(ss)
        if not total_samples:
            return "无数据"
        avg_tokens = total_tokens / total_samples
        # v0：3100 样本、525 token/样本、531 s/epoch → 0.326 µs·epoch⁻¹/token
        seconds = total_tokens * 531 / (3100 * 525)
        return (f"{total_samples} 样本 · 均 {avg_tokens:.0f} token/样本 · "
                f"预计 ~{seconds * epochs / 60:.0f} 分钟 / {epochs} epoch（v0 为 3100×525 → 3×531s）")
    except Exception as e:  # noqa: BLE001 - 估时失败不该挡住流水线
        return f"(估时失败: {e})"


def md_table(rows: list[tuple[str, str]]) -> str:
    return "\n".join(f"| {k} | {v} |" for k, v in rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--i-am-away", action="store_true",
                   help="确认 owner 已离开电脑：会开可见 mock 窗口 + 占满 GPU")
    p.add_argument("--episodes", type=int, default=3000, help="mock v3 回合数")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--base", default="Qwen/Qwen3-0.6B")
    p.add_argument("--ckpt-out", default="checkpoints/v3", help="新 ckpt 输出目录（不动 v0）")
    p.add_argument("--with-v2", action="store_true", help="训练时也掺 mock v2（默认只用 v3+真实）")
    p.add_argument("--skip-gen", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="只走一遍控制流、不真跑子进程（校验参数拼接与报告，可随时用）")
    p.add_argument("--allow-no-v3", action="store_true",
                   help="v3 数据为空时也继续（默认中止：分布对齐是本流水线的目的）")
    p.add_argument("--idle-gate-s", type=float, default=45.0,
                   help="用户空闲满这么多秒才继续（人在用就自动暂停；0=不设闸门）")
    p.add_argument("--throttle-ms", type=int, default=50,
                   help="训练每个优化步让路这么多毫秒（配合 idle-gate）")
    p.add_argument("--eval-episodes", type=int, default=60)
    args = p.parse_args()

    if not args.i_am_away and not args.dry_run:
        raise SystemExit(
            "拒绝运行：会开可见 mock 窗口（零鼠标/零焦点，但窗口真实存在）并占满 GPU。\n"
            "owner 在电脑前时请只跑无头部分（scripts/bench_jev.py / eval_offline.py）。\n"
            "确认可以开窗时加 --i-am-away；只想校验脚本逻辑加 --dry-run。")

    stamp = time.strftime("%Y%m%d-%H%M")
    log_dir = ROOT / "assets" / "bench" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"pipeline_{stamp}.log"
    v3_dir = HOME / "traces_mock_v3"
    report: list[tuple[str, str]] = []

    with open(log_path, "w", encoding="utf-8") as log:
        print(f"流水线开始 {time.strftime('%Y-%m-%d %H:%M:%S')} → {log_path}", flush=True)
        report.append(("开始", time.strftime("%Y-%m-%d %H:%M:%S")))
        _pipeline(args, log, report, stamp, v3_dir)

        md = ROOT / "assets" / "bench" / f"pipeline_{stamp}.md"
        md.write_text(
            f"# 流水线结果 {stamp}\n\n| 项 | 值 |\n|---|---|\n{md_table(report)}\n\n"
            f"日志：`{log_path.relative_to(ROOT)}`\n", encoding="utf-8")
        print(f"\n完成 → {md}\n日志 → {log_path}", flush=True)
        print("\n".join(f"{k}: {v}" for k, v in report), flush=True)


def _pipeline(args, log, report: list, stamp: str, v3_dir: Path) -> None:
    """顺序跑各步骤；任一步失败就停下并如实记进报告。"""
    # ---- 1. mock v3 量产（断点续跑） ----
    if not args.skip_gen:
        have = count_episodes(v3_dir)
        report.append(("mock v3 已有回合", str(have)))
        if have >= args.episodes:
            print("v3 数据已够，跳过量产", flush=True)
        else:
            code = run_step(
                f"mock v3 量产（{have + 1} → {args.episodes}）",
                [PY, str(ROOT / "scripts" / "mock_v3.py"),
                 "--episodes", str(args.episodes), "--seed", str(args.seed),
                 "--start", str(have + 1),
                 "--idle-gate-s", str(args.idle_gate_s)],
                log, dry=args.dry_run)
            report.append(("mock v3 量产", "成功" if code == 0 else f"失败({code})"))
            if code != 0:
                print("量产失败，后面步骤跳过", flush=True)
                return
    report.append(("v3 回合总数", str(count_episodes(v3_dir))))
    if count_episodes(v3_dir) == 0 and not args.allow_no_v3:
        report.append(("中止", "mock v3 数据为空：只吃真实/ v2 数据达不到分布对齐的目的"))
        print("mock v3 数据为空 → 中止（要强行继续加 --allow-no-v3）", flush=True)
        return

    # ---- 2. 训练（输出到独立目录，不覆盖 v0） ----
    data_args: list[str] = []
    for d in ([v3_dir, HOME / "traces_full"]
              + ([HOME / "traces_mock"] if args.with_v2 else [])):
        if d.exists():
            data_args += ["--data", str(d)]
    report.append(("训练数据", " + ".join(Path(a).name for a in data_args[1::2])))
    dirs = [Path(a) for a in data_args[1::2]]
    if dirs:
        estimate = estimate_training(dirs, args.base, args.epochs)
        report.append(("训练时长预估", estimate))
        print(f"训练时长预估：{estimate}", flush=True)
    if not args.skip_train:
        code = run_step(
            f"训练（{args.base}，{args.epochs} epoch）",
            [PY, "-m", "training.train", *data_args, "--base", args.base,
             "--epochs", str(args.epochs), "--out", args.ckpt_out,
             "--idle-gate-s", str(args.idle_gate_s),
             "--throttle-ms", str(args.throttle_ms)],
            log, dry=args.dry_run)
        report.append(("训练", "成功" if code == 0 else f"失败({code})"))
        if code != 0:
            return

    ckpt = latest_ckpt(ROOT / args.ckpt_out, args.epochs)
    if ckpt is None:
        report.append(("新 ckpt", "未找到（训练被跳过或失败）"))
        print("没找到新 ckpt，跳过校准与评测", flush=True)
        return
    report.append(("新 ckpt", str(ckpt.relative_to(ROOT))))

    # ---- 3. 校准（τ 自动选：新分布的 coverage 曲线） ----
    code = run_step("校准 τ + 落盘 isotonic/训练分布统计",
                    [PY, "-m", "training.calibrate", "--ckpt", str(ckpt),
                     *data_args], log, dry=args.dry_run)
    report.append(("校准", "成功" if code == 0 else f"失败({code})"))

    # ---- 4. 真实域复测（北极星的关键：mock 涨了不算数，真实域才算） ----
    real_out = ROOT / "assets" / "bench" / f"v3_realdomain_{stamp}.json"
    run_step("真实域复测（traces_full）",
             [PY, str(ROOT / "scripts" / "eval_offline.py"), "--ckpt", str(ckpt),
              "--data", str(HOME / "traces_full"), "--seeds", "2",
              "--latency-samples", "60", "--out", str(real_out)], log,
             dry=args.dry_run)

    # ---- 5. mock 域基准（确认没把训练域搞坏 + 拿时延） ----
    mock_out = ROOT / "assets" / "bench" / f"v3_mock_{stamp}.json"
    run_step("mock 基准（Jev 单边，省 API）",
             [PY, str(ROOT / "scripts" / "bench_jev.py"), "--ckpt", str(ckpt),
              "--episodes", str(args.eval_episodes), "--skip-vlm",
              "--out", str(mock_out.parent)], log, dry=args.dry_run)

    # ---- 6. 对照表（v0 真实域指标 → v3） ----
    def load(p: Path) -> dict:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    real = load(real_out).get("deployed") or {}
    v0_real = load(ROOT / "assets" / "bench"
                   / "offline_eval_realdomain.json").get("deployed") or {}
    for label, key in (("op_acc", "op_acc"), ("joint_acc", "joint_acc"),
                       ("升级率", "escalation_rate"), ("覆盖率", "coverage")):
        report.append((f"真实域 {label}（v0 → v3）",
                       f"{v0_real.get(key, '?')} → {real.get(key, '?')}"))
    report.append(("新 τ", str(load(ckpt / "calibration.json").get("tau", "?"))))


if __name__ == "__main__":
    main()
