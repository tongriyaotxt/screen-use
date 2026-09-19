"""真 GUI 端到端：mock 窗口 + 真 `VisualLoop` + 零鼠标执行器（owner 离开电脑时才跑）。

和 `bench_jev.py`（无头 replay）的分工：
- replay 测的是「决策耗时」——观测来自落盘轨迹，不需要真窗口；
- 本脚本测的是「真墙钟 + 真实升级率」——真窗口、真 UIA 枚举、真 SoM 截图、真动作执行。

零干扰设计（沿用 mock v2 的三条）：
1. 窗口 `WA_ShowWithoutActivating`（不抢焦点）+ 固定右下角小窗；
2. 所有点击走 UIA Invoke/Select/Toggle/SetFocus，**不动鼠标**（`Executor` 完全不用）；
3. 弹层（菜单/下拉）按 mock_env 的规则开一关一，防 grabMouse 劫持 owner 鼠标；
   `finally` 里强制关窗 + 关弹层。

但它依然会在屏幕角落出现小窗，所以**默认拒绝运行**，必须显式给 `--allow-gui`。

用法（项目根，.venv）：
    # 0) 自检：offscreen 平台，不显示任何窗口（可随时跑）
    ./.venv/Scripts/python.exe scripts/bench_gui.py --offscreen --episodes 1 --driver jev
    # 1) owner 离开时：环境基线（oracle 驾驶，零模型）
    ./.venv/Scripts/python.exe scripts/bench_gui.py --allow-gui --episodes 8 --driver oracle
    # 2) owner 离开时：Jev 混合驾驶 vs 纯 VLM 驾驶
    ./.venv/Scripts/python.exe scripts/bench_gui.py --allow-gui --episodes 8 --driver jev
    ./.venv/Scripts/python.exe scripts/bench_gui.py --allow-gui --episodes 4 --driver vlm
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # mock_env 是同目录脚本

from PIL import Image  # noqa: E402

from screen_use.memory import ExperienceStore  # noqa: E402
from screen_use.perception import screen, som  # noqa: E402
from screen_use.perception.uia_tree import UIElement  # noqa: E402


class MockGuiTools:
    """把 mock 窗口包装成 `VisualLoop` 需要的最小 tools 接口（零鼠标 / 零键盘）。"""

    def __init__(self, win, runner, episode_id: str, provider, settings):
        self.win = win
        self.runner = runner
        self.episode_id = episode_id
        self.settings = settings
        self._provider = provider
        self._elements: list[dict] = []
        self._focused_gid: int | None = None
        self.actions: list[tuple[str, str]] = []      # (动作, 元素名/文本) 供序列比对
        self.timings = {"observe_ms": 0.0, "act_ms": 0.0, "shot_ms": 0.0}
        self._shot_path = Path(tempfile.gettempdir()) / "mock_gui_shot.png"

    # ---- VisualLoop 依赖的接口 ----

    @property
    def provider(self):
        if self._provider is None:
            raise RuntimeError("VLM 未配置：本脚本只在 driver=vlm 或升级时需要")
        return self._provider

    def list_ui_elements(self) -> dict:
        t0 = time.perf_counter()
        els = self.runner._observe(self.episode_id)  # mock_env 的过滤 + id 重排（保留 _gid）
        self.timings["observe_ms"] += (time.perf_counter() - t0) * 1000
        self._elements = els
        return {"foreground_title": f"MockUI-{self.episode_id}",
                "elements": [_public(e) for e in els]}

    def screenshot(self, annotate: bool = False, max_size: int | None = None,
                   quality: int | None = None) -> dict:
        """窗口局部截图（不含 owner 桌面）→ 可选 SoM 标注。

        坐标换算：元素 bbox 是屏幕坐标，图是窗口客户区，标注前先减窗口原点。
        """
        t0 = time.perf_counter()
        els = self.runner._observe(self.episode_id)
        self._elements = els
        img = self._grab_window()
        origin = self._client_origin()
        dpr = img.size[0] / max(1, self.win.width())
        local = [
            UIElement(id=e["id"], name=e["name"], control_type=e["control_type"],
                      bbox=tuple(int((v - (origin[0] if i % 2 == 0 else origin[1])) / dpr)
                                 for i, v in enumerate(e["bbox"])))
            for e in els
        ]
        scale = 1.0
        if annotate:
            img, scale = screen.downscale(img, max_size or self.settings.screenshot_max_size)
            img = som.annotate(img, local, scale)
        b64, used_q = screen.to_jpeg_base64_capped(
            img, quality or self.settings.screenshot_jpeg_quality,
            self.settings.screenshot_max_bytes)
        self.timings["shot_ms"] += (time.perf_counter() - t0) * 1000
        return {"image_base64": b64, "scale": scale * dpr, "width": img.size[0],
                "height": img.size[1], "quality": used_q,
                "foreground_title": f"MockUI-{self.episode_id}",
                "elements": [_public(e) for e in els]}

    def click_element_id(self, element_id: int) -> dict:
        el = next((e for e in self._elements if e["id"] == element_id), None)
        if el is None:
            return {"ok": False, "detail": f"元素 #{element_id} 不在最近一次观测里"}
        t0 = time.perf_counter()
        ok = self.runner._click_target(self.win, el["name"], el)
        self.timings["act_ms"] += (time.perf_counter() - t0) * 1000
        self.actions.append(("click", el["name"]))
        self._after_action(el)
        return {"ok": bool(ok), "element": _public(el),
                "detail": f"点击 {el['name']}" + ("" if ok else "（UIA invoke 失败）")}

    def type_text(self, text: str) -> dict:
        if self._focused_gid is None:
            return {"ok": False, "detail": "没有已聚焦的输入框（先 click_id 目标输入框）"}
        t0 = time.perf_counter()
        r = self.runner.tools.set_element_text(self._focused_gid, text)
        self.timings["act_ms"] += (time.perf_counter() - t0) * 1000
        self.actions.append(("type", text))
        return r | {"detail": f"写入 {text!r}（UIA SetValue，零键盘）"}

    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> dict:
        t0 = time.perf_counter()
        sb = self.win.listbox.verticalScrollBar() if self.win.listbox is not None else None
        if sb is not None:
            sb.setValue(min(sb.maximum(), sb.value() + 100 * abs(int(clicks))))
        self.timings["act_ms"] += (time.perf_counter() - t0) * 1000
        self.actions.append(("scroll", str(clicks)))
        return {"ok": sb is not None, "detail": f"滚动 {clicks}（Qt 滚动条直驱）"}

    def double_click(self, x: int, y: int) -> dict:
        return self._coord_fallback("double_click", x, y)

    def right_click(self, x: int, y: int) -> dict:
        return self._coord_fallback("right_click", x, y)

    def click(self, x: int, y: int) -> dict:
        return self._coord_fallback("click", x, y)

    def press(self, key: str) -> dict:
        return {"ok": False, "detail": f"零键盘策略：不支持 press({key})"}

    def hotkey(self, *keys: str) -> dict:
        return {"ok": False, "detail": f"零键盘策略：不支持 hotkey({'+'.join(keys)})"}

    # ---- 内部：坐标兜底 / 弹层与焦点管理 ----

    def _coord_fallback(self, kind: str, x: int, y: int) -> dict:
        """坐标点击兜底：换算回屏幕坐标后命中元素表里的控件，仍走 UIA（不动鼠标）。"""
        sx = int(x) + self._client_origin()[0]
        sy = int(y) + self._client_origin()[1]
        hit = next((e for e in self._elements
                    if e["bbox"][0] <= sx <= e["bbox"][2]
                    and e["bbox"][1] <= sy <= e["bbox"][3]), None)
        if hit is None:
            return {"ok": False, "detail": f"({sx},{sy}) 没有可点元素（坐标点击不做鼠标）"}
        self.actions.append((kind, hit["name"]))
        ok = self.runner._click_target(self.win, hit["name"], hit)
        self._after_action(hit)
        return {"ok": bool(ok), "detail": f"{kind} → {hit['name']}"}

    def _after_action(self, el: dict) -> None:
        """动作后处理：记焦点（type 用）、关弹层（防 grabMouse 劫持 owner 鼠标）。"""
        from PySide6.QtWidgets import QApplication

        if el["control_type"] == "EditControl":
            self._focused_gid = el["_gid"]
        combo_name = (self.win.spec.get("combo") or {}).get("name")
        opened_popup = el["name"] in getattr(self.win, "menus", {}) or el["name"] == combo_name
        if not opened_popup:
            self.runner._close_popups(self.win)
            self._focused_gid = None if el["control_type"] != "EditControl" else self._focused_gid
        QApplication.processEvents()
        time.sleep(0.15)

    def _client_origin(self) -> tuple[int, int]:
        from PySide6.QtCore import QPoint

        pt = self.win.mapToGlobal(QPoint(0, 0))
        return int(pt.x()), int(pt.y())

    def _grab_window(self) -> Image.Image:
        """Qt 自己渲染窗口 → PIL（不截 owner 桌面，也不动鼠标）。"""
        self.win.grab().save(str(self._shot_path), "PNG")
        return Image.open(self._shot_path).convert("RGB")

    def cleanup(self) -> None:
        try:
            self.runner._close_popups(self.win)
        except Exception:  # noqa: BLE001 - 清理失败不掩盖主流程结果
            pass

    def oracle_act(self, action: str, target: str | None, text: str | None) -> None:
        """oracle 基线：把模板动作翻译成元素表中的控件名再执行（零决策）。"""
        els = self.runner._observe(self.episode_id)
        self._elements = els
        if target == "__LAST_ITEM__":
            target = self.win.list_items[-1]
        if action == "click_id" and target:
            el = next((e for e in els if e["name"] == target), None)
            if el is not None:
                self.runner._click_target(self.win, target, el)
                self.actions.append(("click", target))
                self._after_action(el)
        elif action == "type" and target:
            el = next((e for e in els if e["name"] == target), None)
            if el is not None:
                self.runner.tools.set_element_text(el["_gid"], text or "")
                self.actions.append(("type", text or ""))
        elif action == "scroll":
            sb = self.win.listbox.verticalScrollBar() if self.win.listbox is not None else None
            if sb is not None:
                sb.setValue(min(sb.maximum(), sb.value() + 100 * abs(int(text or 0))))
                self.actions.append(("scroll", str(text)))


def _public(el: dict) -> dict:
    return {k: v for k, v in el.items() if k != "_gid"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--driver", choices=["oracle", "jev", "vlm"], default="jev")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", default="checkpoints/jev-Qwen3-0.6B-e3")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--step-delay", type=float, default=0.15)
    p.add_argument("--template", default="", help="只跑指定范式（如 t_combo_select）")
    p.add_argument("--allow-gui", action="store_true", help="确认 owner 已离开电脑")
    p.add_argument("--offscreen", action="store_true",
                   help="自检模式：QT_QPA_PLATFORM=offscreen，不显示任何窗口（UIA 看不到窗口）")
    p.add_argument("--json-out", default="")
    args = p.parse_args()

    if not args.allow_gui and not args.offscreen:
        raise SystemExit(
            "拒绝运行：本脚本会开可见的 mock 小窗（零鼠标/零焦点，但窗体会出现在右下角）。\n"
            "owner 在电脑前时请用 scripts/bench_jev.py（无头 replay）；\n"
            "确认可以开窗时加 --allow-gui；只想自检代码路径加 --offscreen。")

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    import mock_env

    home = Path(tempfile.mkdtemp(prefix="bench_gui_"))
    app = QApplication.instance() or QApplication(sys.argv)

    jev = None
    if args.driver == "jev":
        from screen_use.jev_bridge import load_engine

        jev = load_engine(args.ckpt)
        if jev is None:
            raise SystemExit(f"Jev 加载失败：{args.ckpt}")

    provider = None
    if args.driver == "vlm" or (args.driver == "jev" and not args.offscreen):
        from screen_use.config import Settings
        from screen_use.providers.openai_compat import OpenAICompatProvider

        settings = Settings()
        if settings.vlm_available:
            provider = OpenAICompatProvider(settings)

    from screen_use.config import Settings
    from screen_use.agent import VisualLoop

    settings = Settings(_env_file=None, record_full_trace=False)
    rng = random.Random(args.seed)
    templates = [t for t, _ in mock_env.TEMPLATES]
    if args.template:
        templates = [t for t in templates if t.__name__ == args.template] or templates

    rows = []
    for idx in range(1, args.episodes + 1):
        template = mock_env._WEIGHTED[(idx - 1) % len(mock_env._WEIGHTED)]
        if args.template and template not in templates:
            template = templates[(idx - 1) % len(templates)]
        spec, goal, oracle_steps = template(rng)
        episode_id = f"gui-{args.driver}-{idx:03d}"
        win = mock_env.MockWindow(rng, episode_id, spec)
        win.show()
        app.processEvents()
        time.sleep(0.5)
        runner = mock_env.EpisodeRunner(home / "traces")
        tools = MockGuiTools(win, runner, episode_id, provider, settings)
        t0 = time.perf_counter()
        try:
            if args.driver == "oracle":
                for action, target, text in oracle_steps:
                    tools.oracle_act(action, target, text)
                    app.processEvents()
                    time.sleep(0.15)
                done, steps, ring = True, len(oracle_steps), "oracle"
                metrics = {"wall_ms": (time.perf_counter() - t0) * 1000}
            else:
                loop = VisualLoop(tools, max_steps=args.max_steps,
                                  step_delay=args.step_delay,
                                  memory=ExperienceStore(home / "mem"), jev=jev)
                result = loop.run(goal)
                done, ring = result.done, result.metrics["ring"]
                metrics, steps = result.metrics, len(result.steps)
        finally:
            tools.cleanup()
            win.close()
            app.processEvents()

        wall = (time.perf_counter() - t0) * 1000
        expected = [(a, t) for a, t, _ in oracle_steps if a in ("click_id", "type", "scroll")]
        got = list(tools.actions)
        matched = sum(1 for i, e in enumerate(expected) if i < len(got) and _same(got[i], e))
        rows.append({
            "episode": idx, "template": template.__name__, "goal": goal,
            "done": done, "ring": ring, "steps": steps,
            "wall_ms": round(wall, 1),
            "observe_ms": round(tools.timings["observe_ms"], 1),
            "shot_ms": round(tools.timings["shot_ms"], 1),
            "act_ms": round(tools.timings["act_ms"], 1),
            "metrics": metrics,
            "oracle_steps": len(expected), "oracle_matched": matched,
            "actions": got, "expected": [f"{a}:{t}" for a, t in expected],
        })
        print(f"[{idx}/{args.episodes}] {template.__name__} done={done} ring={ring} "
              f"wall={wall:.0f}ms 步={steps} 对齐={matched}/{len(expected)}", flush=True)

    summary = {
        "driver": args.driver, "episodes": len(rows), "offscreen": args.offscreen,
        "done_rate": round(sum(1 for r in rows if r["done"]) / max(len(rows), 1), 3),
        "oracle_align": round(sum(r["oracle_matched"] for r in rows)
                              / max(sum(r["oracle_steps"] for r in rows), 1), 3),
        "wall_ms_mean": round(statistics.mean([r["wall_ms"] for r in rows]), 1),
        "vlm_calls": sum((r["metrics"] or {}).get("vlm_calls", 0) for r in rows),
        "jev_steps": sum((r["metrics"] or {}).get("jev_steps", 0) for r in rows),
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                     ensure_ascii=False, indent=1))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ {out}")


def _same(got: tuple[str, str], expect: tuple[str, str]) -> bool:
    """动作类别与目标是否对齐（type 比文本，click 比元素名）。"""
    kind = "click" if expect[0] == "click_id" else expect[0]
    return got[0] == kind and got[1] == expect[1]


if __name__ == "__main__":
    main()
