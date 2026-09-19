"""集成冒烟：真实桌面上跑一次 Jev 快路径（零鼠标 / 零焦点抢占 / 零外发 / 零落盘）。

为什么需要它：`bench_jev.py` 是离线 replay（观测来自轨迹），验证不了
「VisualLoop 里的 Jev 分支真的接对了、真桌面上不会炸」。本脚本在真实桌面跑一遍
`run_task` 的完整路径：

- 观测：真实 UIA 枚举 + 真实截图（只在本机内存里，**不外发**——provider 是桩）
- 决策：真实 Jev 引擎（快路径）+ 桩 VLM（慢路径不会发网络请求）
- 执行：dry_run 只记录不真点，并把 `_activate_window` 打成空操作——
  否则 dry_run 也会抢焦点（ShowWindow/SetForegroundWindow 是真的 Win32 调用）
- 落盘：强制 record_full_trace=False——dry_run 的「成功」步不是真轨迹，
  写进 traces_full 会污染 SFT 数据

用法（项目根，.venv）：
    ./.venv/Scripts/python.exe scripts/smoke_jev_loop.py --max-steps 3
    ./.venv/Scripts/python.exe scripts/smoke_jev_loop.py --goal "点击开始按钮" --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from screen_use.agent import VisualLoop  # noqa: E402
from screen_use.config import Settings  # noqa: E402
from screen_use.providers.base import VisionProvider  # noqa: E402
from screen_use.tools import ScreenUse  # noqa: E402


class StubProvider(VisionProvider):
    """不出网的假 VLM：记录收到的 prompt，按脚本返回动作（验证慢路径接线）。"""

    def __init__(self, script: list[str]):
        self.script = script
        self.prompts: list[str] = []

    def pick_element(self, image_base64, legend, goal):
        return -1

    def ask_about_screen(self, image_base64, question):
        self.prompts.append(question)
        return self.script[min(len(self.prompts) - 1, len(self.script) - 1)]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/jev-Qwen3-0.6B-e3")
    p.add_argument("--goal", default="点击「确定」按钮")
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument("--vlm-action", default='{"action": "done", "result": "冒烟：桩 VLM 收尾"}')
    p.add_argument("--json", action="store_true", help="只输出 JSON 结果")
    args = p.parse_args()

    settings = Settings(_env_file=None, jev_checkpoint=args.ckpt,
                        record_full_trace=False, vlm_max_size=896)
    provider = StubProvider([args.vlm_action])
    tools = ScreenUse(dry_run=True, settings=settings, provider=provider)
    tools._activate_window = lambda el: None  # 零焦点抢占（否则 dry_run 也会激活窗口）

    if tools.jev is None:
        raise SystemExit(f"Jev 未加载成功：{args.ckpt}（看上面的 warning 日志）")

    steps_log: list[str] = []

    def on_step(step):
        line = (f"  #{step.index} [{step.ring}] {step.action} {step.args} "
                f"{'✓' if step.ok else '✗'} {step.detail}")
        steps_log.append(line)
        if not args.json:
            print(line, flush=True)

    if not args.json:
        print(f"冒烟开始：goal={args.goal!r} max_steps={args.max_steps}")
        print("（dry_run：不点鼠标不敲键盘；UIA 枚举与截图是本机只读操作）")

    t0 = time.perf_counter()
    result = VisualLoop(tools, max_steps=args.max_steps, jev=tools.jev).run(
        args.goal, on_step=on_step)
    payload = result.to_dict() | {"providers_called": len(provider.prompts),
                                  "elapsed_s": round(time.perf_counter() - t0, 2)}

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=1))
    else:
        m = result.metrics
        print(f"\n结果：done={result.done} ring={m['ring']} 步数={len(result.steps)}")
        print(f"指标：jev_steps={m['jev_steps']} vlm_steps={m['vlm_steps']} "
              f"升级={m['jev_escalations']} 引擎异常={m['jev_errors']} "
              f"截图={m['screenshot_calls']}")
        print(f"耗时：wall={m['wall_ms']:.0f}ms observe={m['observe_ms']:.0f}ms "
              f"jev={m['jev_ms']:.0f}ms vlm={m['vlm_ms']:.0f}ms act={m['act_ms']:.0f}ms")
        print(f"桩 VLM 调用 {len(provider.prompts)} 次（真实 API 调用 0 次）")
        for prompt_hint in ("快路径提示",):
            hit = sum(1 for q in provider.prompts if prompt_hint in q)
            print(f"升级提示注入 {hit}/{len(provider.prompts)} 次")


if __name__ == "__main__":
    main()
