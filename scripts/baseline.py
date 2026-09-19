"""北极星基线工具：同一套任务上给「人」和「agent」计时，产出对照表。

北极星指标是「端到端耗时 vs 人类基线」（cascade-protocol §6 / DESIGN §9），但在本脚本之前
**人类基线根本没被定义过**——没有这个数就不能说"快过人类"。这里把口径固定下来：

- 任务集：8 个日常 GUI 任务（记事本 / 计算器 / 资源管理器 / 浏览器），每个都有**可验证的完成条件**；
- 计时口径：从"看到任务文字"到"自己认为做完"（人）；agent 侧同 goal、同验证器、同口径；
- 验证：文件类任务程序化验证（不看自述），其余靠自述 + 结果截图留档。

三种模式：
    ./.venv/Scripts/python.exe scripts/baseline.py --mode human        # 给人计时（交互，几分钟）
    ./.venv/Scripts/python.exe scripts/baseline.py --mode agent --i-am-away   # 给 agent 计时（要开窗）
    ./.venv/Scripts/python.exe scripts/baseline.py --mode report       # 对照表

产出：assets/bench/baseline_human.json / baseline_agent.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "assets" / "bench"
WORK = Path(os.environ.get("TEMP", ".")) / "screenuse_baseline"


def _p(name: str) -> Path:
    return WORK / name


TASKS: list[dict] = [
    {
        "id": "notepad_type_save",
        "goal": "打开记事本，输入「TODO 周五交报告」，保存到 "
                f"{_p('todo.txt')}",
        "verify": lambda: _p("todo.txt").exists() and
                          "TODO" in _p("todo.txt").read_text(encoding="utf-8",
                                                             errors="ignore"),
        "verify_desc": "文件存在且含 TODO",
    },
    {
        "id": "notepad_edit_existing",
        "goal": f"把 {_p('todo.txt')} 里的内容改成「已完成」并保存",
        "setup": lambda: _p("todo.txt").write_text("TODO 周五交报告", encoding="utf-8"),
        "verify": lambda: _p("todo.txt").exists() and
                          "已完成" in _p("todo.txt").read_text(encoding="utf-8",
                                                               errors="ignore"),
        "verify_desc": "文件内容被改成「已完成」",
    },
    {
        "id": "explorer_new_folder",
        "goal": f"在资源管理器里新建文件夹 {_p('新项目')}",
        "verify": lambda: _p("新项目").is_dir(),
        "verify_desc": "目录存在",
    },
    {
        "id": "explorer_rename",
        "goal": f"把文件夹 {_p('新项目')} 重命名为 {_p('已归档')}",
        "setup": lambda: _p("新项目").mkdir(exist_ok=True),
        "verify": lambda: _p("已归档").is_dir(),
        "verify_desc": "重命名后的目录存在",
    },
    {
        "id": "calc_multiply",
        "goal": "用计算器算出 78×9，把结果写在记事本里保存为 " + f"{_p('calc.txt')}",
        "verify": lambda: _p("calc.txt").exists() and
                          "702" in _p("calc.txt").read_text(encoding="utf-8",
                                                            errors="ignore"),
        "verify_desc": "文件含正确结果 702",
    },
    {
        "id": "browser_search",
        "goal": "在浏览器里搜索「screen-use github」，打开第一个结果",
        "verify": None,          # 浏览内容不做程序化验证，靠自述
        "verify_desc": "自述完成（浏览器状态不做程序判定）",
    },
    {
        "id": "notepad_batch_lines",
        "goal": f"在记事本里输入三行：项目Alpha / 项目Beta / 项目Gamma，保存为 {_p('projects.txt')}",
        "verify": lambda: all(k in _p("projects.txt").read_text(encoding="utf-8",
                                                                errors="ignore")
                              for k in ("项目Alpha", "项目Beta", "项目Gamma"))
        if _p("projects.txt").exists() else False,
        "verify_desc": "三行都在文件里",
    },
    {
        "id": "cleanup_workspace",
        "goal": f"把 {WORK} 目录里的 txt 文件全部删掉",
        "verify": lambda: not list(WORK.glob("*.txt")),
        "verify_desc": "目录里没有 txt 了",
    },
]


def run_human(tasks: list[dict]) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    results = []
    print("人类基线计时：看到任务后按 Enter 开始，自认为做完再按 Enter。")
    print("（中途想跳过按 s + Enter；任务做完请自己保存文件）\n")
    for t in tasks:
        print("=" * 70)
        print(f"[{t['id']}] {t['goal']}")
        print(f"完成判据：{t['verify_desc']}")
        if t.get("setup"):
            t["setup"]()
        cmd = input("准备就绪按 Enter 开始（s=跳过）... ").strip().lower()
        if cmd == "s":
            continue
        t0 = time.perf_counter()
        input(">>> 计时中，做完按 Enter <<<")
        cost = time.perf_counter() - t0
        ok = bool(t["verify"]()) if t["verify"] else input(
            "自动判据不适用，你认为完成了吗？(y/n) ").strip().lower().startswith("y")
        print(f"用时 {cost:.1f}s · 判据{'通过' if ok else '未通过'}\n")
        results.append({"id": t["id"], "goal": t["goal"], "seconds": round(cost, 1),
                        "verified": bool(ok), "verify_desc": t["verify_desc"]})
    out = BENCH / "baseline_human.json"
    BENCH.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "tasks": results}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    ok = [r["seconds"] for r in results if r["verified"]]
    print(f"完成 {len(results)} 个任务，验证通过 {len(ok)} 个 → {out}")
    if ok:
        print(f"人类中位用时 {statistics.median(ok):.1f}s（仅统计验证通过的）")


def run_agent(tasks: list[dict], max_steps: int, use_jev: bool) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    from screen_use.config import Settings
    from screen_use.tools import ScreenUse

    settings = Settings()
    if not use_jev:
        settings.jev_checkpoint = ""
    tools = ScreenUse(settings=settings)
    results = []
    for t in tasks:
        if t.get("setup"):
            t["setup"]()
        print(f"[{t['id']}] {t['goal']}", flush=True)
        t0 = time.perf_counter()
        try:
            r = tools.run_task(t["goal"], max_steps=max_steps)
        except Exception as e:  # noqa: BLE001 - 单个任务失败不该中断整轮
            r = {"status": "error", "result": str(e)}
        cost = time.perf_counter() - t0
        ok = bool(t["verify"]()) if t["verify"] else r.get("status") == "done"
        print(f"  {r.get('status')} · {cost:.1f}s · 判据{'通过' if ok else '未通过'} · "
              f"engine={r.get('engine', '?')}", flush=True)
        results.append({
            "id": t["id"], "goal": t["goal"], "seconds": round(cost, 1),
            "verified": bool(ok), "status": r.get("status"),
            "engine": r.get("engine"), "vlm_calls": (r.get("metrics") or {}).get("vlm_calls"),
            "jev_steps": (r.get("metrics") or {}).get("jev_steps"),
            "steps": len(r.get("steps", [])),
        })
    out = BENCH / "baseline_agent.json"
    BENCH.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "max_steps": max_steps, "jev": use_jev,
                               "tasks": results}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"→ {out}")


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run_report() -> None:
    human = load(BENCH / "baseline_human.json").get("tasks", [])
    agent = load(BENCH / "baseline_agent.json").get("tasks", [])
    by_id_h = {t["id"]: t for t in human}
    by_id_a = {t["id"]: t for t in agent}
    lines = ["# 北极星基线：人 vs agent（同一任务集、同一验证器）", "",
             "| 任务 | 人类(s) | agent(s) | 倍数 | 人类判据 | agent 判据 |",
             "|---|---|---|---|---|---|"]
    ratios = []
    for t in TASKS:
        h, a = by_id_h.get(t["id"]), by_id_a.get(t["id"])
        if not h and not a:
            continue
        hs = f"{h['seconds']}" if h else "-"
        as_ = f"{a['seconds']}" if a else "-"
        ratio = "-"
        if h and a and a["seconds"]:
            r = h["seconds"] / a["seconds"]
            ratios.append(r)
            ratio = f"{r:.2f}×"
        lines.append(f"| {t['goal'][:40]} | {hs} | {as_} | {ratio} | "
                     f"{'✓' if h and h['verified'] else '✗' if h else '-'} | "
                     f"{'✓' if a and a['verified'] else '✗' if a else '-'} |")
    if ratios:
        lines += ["", f"中位倍数 **{statistics.median(ratios):.2f}×**"
                      f"（>1 = agent 更快；这是北极星的单一数字）"]
    md = BENCH / "baseline_report.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n→ {md}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["human", "agent", "report"], required=True)
    p.add_argument("--only", default="", help="只跑这些 id（逗号分隔）")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--no-jev", action="store_true", help="agent 模式：强制纯 VLM 对照")
    p.add_argument("--i-am-away", action="store_true",
                   help="agent 模式需要：会真的操作鼠标键盘")
    args = p.parse_args()

    tasks = [t for t in TASKS
             if not args.only or t["id"] in args.only.split(",")]
    if args.mode == "human":
        run_human(tasks)
    elif args.mode == "agent":
        if not args.i_am_away:
            raise SystemExit("agent 模式会真的操作鼠标键盘，owner 在电脑前时不要跑；"
                             "确认离开加 --i-am-away")
        run_agent(tasks, args.max_steps, not args.no_jev)
    else:
        run_report()


if __name__ == "__main__":
    main()
