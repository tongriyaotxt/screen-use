"""desktop-jev 阶段 4：北极星指标首测 —— 「Jev 驾驶 vs VLM 驾驶」端到端速度对比。

**为什么用 replay（无头）而不是真 GUI**：mock GUI 量产会开窗、弹层还会抓鼠标，
owner 在机时禁跑（PROGRESS 环境备忘）。而速度对比的自变量是「每步决策耗时」，
观测分布可以拿已落盘的 gold 轨迹（~3100 步 mock）原样复现：同一步、同一元素表、
同一 goal，分别喂给两个驾驶者——比合成新场景更公平（同一批观测，配对比较）。

**测什么**（TRAINING §7 在线指标的无头版）：
1. 决策时延 p50/p95 —— Jev 单次前向 vs VLM 一次 chat 往返（含纠错重试）
2. 决策质量 —— op_acc / joint_acc（对 oracle 标签，teacher forcing）
3. 快路径覆盖率 / 升级率（serve 侧校准后的 conf 与 τ）
4. 端到端推算 —— 每任务 Σ(观测 + 决策 + 执行)；观测/执行用本机实测常量，
   对照组 = 纯 VLM 驾驶（现状）

**VLM 侧看到的图**：底图是合成的（灰白窗口 + 控件名文字），但**编号框用的就是
真实 `som.annotate`**、图例与 prompt 与线上 `VisualLoop` 逐字符一致——因为 VLM
在这个循环里的任务是「看图例选编号」，不是读像素。合成底图会让 VLM 略微吃亏，
报告里如实标注。

用法（项目根，.venv；Windows 控制台 GBK → 用 PYTHONIOENCODING=utf-8）：
    ./.venv/Scripts/python.exe scripts/bench_jev.py --episodes 40 --skip-vlm
    ./.venv/Scripts/python.exe scripts/bench_jev.py --episodes 40 --vlm-episodes 3
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # training/ 未安装进 venv

from screen_use import jev_bridge  # noqa: E402
from screen_use.agent import _LOOP_PROMPT, parse_actions  # noqa: E402  (prompt 与线上循环同源)
from screen_use.config import get_settings  # noqa: E402
from screen_use.memory import _default_home  # noqa: E402
from screen_use.perception import screen, som  # noqa: E402
from screen_use.perception.uia_tree import UIElement  # noqa: E402
from screen_use.tools import ScreenUse  # noqa: E402
from training.common import load_trace_files  # noqa: E402

DESKTOP_SIZE = (1280, 800)  # mock 轨迹采集时的工作分辨率（逻辑=物理像素）
_FONTS = (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf")


# ==================== 渲染（合成底图 + 真实 SoM 标注） ====================

def _font(size: int):
    for path in _FONTS:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _clamp_rect(bbox, width: int, height: int, pad_l: int = 0, pad_t: int = 0,
                pad_r: int = 0, pad_b: int = 0):
    """bbox → 画布内合法矩形（元素可能在窗口外/屏幕外，PIL 对越界坐标会报错）。"""
    left = min(max(0, int(bbox[0]) - pad_l), width - 1)
    top = min(max(0, int(bbox[1]) - pad_t), height - 1)
    right = max(min(width, int(bbox[2]) + pad_r), left + 1)
    bottom = max(min(height, int(bbox[3]) + pad_b), top + 1)
    return [left, top, right, bottom]


def render_som(elements: list[dict], max_size: int = 896):
    """元素表 → (SoM 图, scale)。编号框/图例走线上同一份代码。"""
    width, height = DESKTOP_SIZE
    img = Image.new("RGB", DESKTOP_SIZE, (245, 246, 248))
    draw = ImageDraw.Draw(img)
    font, title_font = _font(12), _font(13)

    groups: dict[str, list[dict]] = {}
    for el in elements:
        groups.setdefault(el.get("window_title") or "(无标题)", []).append(el)
    for title, els in groups.items():
        xs = [e["bbox"][0] for e in els] + [e["bbox"][2] for e in els]
        ys = [e["bbox"][1] for e in els] + [e["bbox"][3] for e in els]
        left, top, right, bottom = _clamp_rect(
            [min(xs), min(ys), max(xs), max(ys)], width, height, 20, 30, 20, 20)
        draw.rectangle([left, top, right, bottom], fill=(255, 255, 255),
                       outline=(176, 180, 186))
        draw.rectangle([left, top, right, top + 22], fill=(238, 240, 243))
        draw.text((left + 8, top + 4), title, fill=(40, 44, 52), font=title_font)
    for el in elements:
        left, top, right, bottom = _clamp_rect(el["bbox"], width, height)
        draw.rectangle([left, top, right, bottom], outline=(205, 208, 214))
        name = (el.get("name") or "")[:24]
        if name:
            draw.text((left + 6, (top + bottom) // 2 - 8), name, fill=(28, 30, 36),
                      font=font)

    scaled, scale = screen.downscale(img, max_size)
    boxes = [
        UIElement(id=e["id"], name=e.get("name", ""),
                  control_type=e.get("control_type", ""), bbox=tuple(e["bbox"]))
        for e in elements
    ]
    return som.annotate(scaled, boxes, scale), scale


def to_b64(img: Image.Image, quality: int = 80) -> str:
    return base64.b64encode(screen.to_jpeg_bytes(img, quality)).decode("ascii")


# ==================== 轨迹 → 回合 ====================

def oracle_op(action: str) -> str | None:
    return jev_bridge.ACTION_TO_JEV.get(action)


def oracle_target(step: dict) -> int | None:
    args = step.get("args", {})
    return args.get("id") or args.get("element_id")


def load_episodes(mock_dir: Path, limit: int, max_steps: int) -> list[dict]:
    """gold 轨迹 → 回合列表（只取 oracle 全成功的干净回合）。

    抽样用**等距跨整个目录**，不是取前 N 个：mock 目录里前 656 个文件是 v1
    合成器（6 类简单范式）、后面才是 v2（15 类语义/顺序范式），取前缀会系统性
    偏简单，把两个驾驶者都测成"高分"。
    """
    pool = []
    for path, steps in load_trace_files([mock_dir]):
        if not steps or any(not s.get("ok", True) for s in steps):
            continue
        if len(steps) < 2:
            continue
        pool.append({
            "id": path.stem,
            "goal": steps[0]["goal"],
            "steps": steps[:max_steps],
            "v2": path.stem.startswith("mock-v2"),
        })
    if not pool:
        return []
    stride = max(1, len(pool) // limit)
    return pool[::stride][:limit]


# ==================== 两个驾驶者 ====================

def drive_jev(jev, episode: dict) -> dict:
    """Jev 驾驶一遍（teacher forcing：history 用 oracle 动作，保证单步可比）。"""
    recs, history = [], []
    for step in episode["steps"]:
        expected = oracle_op(step["action"])
        t0 = time.perf_counter()
        try:
            d = jev.decide(episode["goal"], step.get("foreground_title", ""),
                           step.get("elements", []), list(history[-4:]))
            err = None
        except Exception as e:  # noqa: BLE001 - 引擎异常按升级处理（与线上一致）
            d = {"action": "ERROR", "element_id": None, "confidence": 0.0,
                 "escalate": True, "latency_ms": 0.0}
            err = str(e)
        ms = float(d.get("latency_ms") or (time.perf_counter() - t0) * 1000)

        target = oracle_target(step)
        op_ok = d["action"] == expected
        elem_ok = (d.get("element_id") == target) if target is not None else (
            d.get("element_id") is None)
        text_ok = True
        if expected == "TYPE":
            text_ok = d.get("text") == step.get("args", {}).get("text")
        recs.append({
            "expected": expected, "pred": d["action"], "escalate": bool(d["escalate"]),
            "op_ok": op_ok, "elem_ok": elem_ok, "text_ok": text_ok,
            "joint_ok": bool(op_ok and elem_ok and text_ok),
            "conf": float(d.get("confidence") or 0.0), "ms": ms, "error": err,
            "in_action_set": expected is not None,
            "blocked_hit": d["action"] == "BLOCKED" and expected == "BLOCKED",
            "blocked_false": d["action"] == "BLOCKED" and expected != "BLOCKED",
        })
        history.append(jev_bridge.history_entry(step["action"], step.get("args", {}),
                                                bool(step.get("ok", True))))
    return {"id": episode["id"], "recs": recs}


def vlm_think(provider, image_b64: str, prompt: str) -> tuple[list[dict] | None, float, int]:
    """复刻 VisualLoop._think：解析失败带错误反馈重试 1 次。返回 (动作, 总耗时, 尝试次数)。"""
    total, attempts, last = 0.0, 0, None
    current = prompt
    for _ in range(2):
        t0 = time.perf_counter()
        try:
            raw = provider.ask_about_screen(image_b64, current)
        except Exception as e:  # noqa: BLE001 - 网络/服务端错误也算这一轮的代价
            total += (time.perf_counter() - t0) * 1000
            attempts += 1
            last = e
            continue
        total += (time.perf_counter() - t0) * 1000
        attempts += 1
        try:
            return parse_actions(raw), total, attempts
        except ValueError as e:
            last = e
            current = prompt + (
                f"\n\n【纠错】你上次的输出无法解析为动作 JSON：{e}\n"
                "请只输出一个 JSON 对象，不要输出任何其他内容。"
            )
    return None, total, attempts


def drive_vlm(provider, episode: dict, max_size: int) -> dict:
    """VLM 驾驶一遍（同一步、同一元素表；图是合成底图 + 真实 SoM 框）。"""
    recs, history = [], []
    for step in episode["steps"]:
        elements = step.get("elements", [])
        img, scale = render_som(elements, max_size)
        legend = "\n".join(
            f"{el['id']}: [{el['window_title']}] [{el['control_type']}] "
            f"{el['name'] or '(无名)'}" for el in elements) or "(无可交互元素)"
        prompt = _LOOP_PROMPT.format(
            goal=episode["goal"], legend=legend,
            history="\n".join(history) if history else "(无)",
            foreground=step.get("foreground_title", ""),
            experience="", reflection="",
            width=img.size[0], height=img.size[1], scale=scale,
        )
        actions, ms, attempts = vlm_think(provider, to_b64(img), prompt)

        expected = oracle_op(step["action"])
        target = oracle_target(step)
        pred, elem_pred, text_ok = None, None, True
        if actions:
            first = actions[0]
            pred = jev_bridge.ACTION_TO_JEV.get(first.get("action"))
            if first.get("action") == "click_id":
                elem_pred = first.get("id")
            elif first.get("action") == "type":
                elem_pred = first.get("id")
                text_ok = first.get("text") == step.get("args", {}).get("text")
        else:
            text_ok = False
        elem_ok = (elem_pred == target) if target is not None else (elem_pred is None)
        op_ok = pred == expected
        recs.append({
            "expected": expected, "pred": pred, "ms": ms, "attempts": attempts,
            "op_ok": op_ok, "elem_ok": elem_ok, "text_ok": text_ok,
            "joint_ok": bool(op_ok and elem_ok and text_ok),
            "in_action_set": expected is not None, "parsed": actions is not None,
        })
        history.append(f"第{len(recs)}步: {step['action']} {step.get('args', {})} -> 成功")
    return {"id": episode["id"], "recs": recs}


# ==================== 统计 ====================

def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1))))
    return ordered[idx]


def summarize_jev(pairs: list[dict]) -> dict:
    recs = [r for p in pairs for r in p["recs"]]
    inset = [r for r in recs if r["in_action_set"]]
    lat = [r["ms"] for r in recs]
    return {
        "steps": len(recs), "steps_in_action_set": len(inset),
        "steps_out_of_action_set": len(recs) - len(inset),
        "op_acc": _rate(inset, "op_ok"), "joint_acc": _rate(inset, "joint_ok"),
        "elem_acc": _rate(inset, "elem_ok"), "type_text_acc": _rate(
            [r for r in inset if r["expected"] == "TYPE"], "text_ok"),
        "escalation_rate": _rate(recs, "escalate"),
        "coverage": 1 - _rate(recs, "escalate"),
        "blocked_hit": sum(1 for r in recs if r["blocked_hit"]),
        "blocked_false": sum(1 for r in recs if r["blocked_false"]),
        "errors": sum(1 for r in recs if r["error"]),
        "latency_ms": _latency(lat),
    }


def summarize_vlm(pairs: list[dict]) -> dict:
    recs = [r for p in pairs for r in p["recs"]]
    inset = [r for r in recs if r["in_action_set"]]
    return {
        "steps": len(recs), "steps_in_action_set": len(inset),
        "op_acc": _rate(inset, "op_ok"), "joint_acc": _rate(inset, "joint_ok"),
        "elem_acc": _rate(inset, "elem_ok"),
        "parse_fail": sum(1 for r in recs if not r["parsed"]),
        "attempts": sum(r["attempts"] for r in recs),
        "latency_ms": _latency([r["ms"] for r in recs]),
    }


def _rate(recs: list[dict], key: str) -> float:
    return round(sum(1 for r in recs if r[key]) / len(recs), 4) if recs else 0.0


def _latency(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "n": 0}
    return {"mean": round(statistics.mean(values), 1), "p50": round(pct(values, 50), 1),
            "p95": round(pct(values, 95), 1), "max": round(max(values), 1),
            "n": len(values)}


def measure_observe(tools: ScreenUse, samples: int = 3) -> dict:
    """本机实测：快路径观测（UIA 枚举）vs 慢路径观测（截图+SoM+编码）。

    只测耗时，截图不落盘、不外发（就在内存里等着被 GC）。
    """
    light, full = [], []
    for _ in range(samples):
        t0 = time.perf_counter()
        tools.list_ui_elements()
        light.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        tools.screenshot(annotate=True, max_size=tools.settings.vlm_max_size)
        full.append((time.perf_counter() - t0) * 1000)
    return {"jev_observe_ms": round(statistics.median(light), 1),
            "vlm_observe_ms": round(statistics.median(full), 1),
            "samples": samples}


def e2e(jev_pairs: list[dict], vlm_pairs: list[dict], obs: dict, act_ms: float,
        max_steps: int) -> dict:
    """配对推算端到端：同一回合、同一步，两个驾驶者各算一遍墙钟时间。"""
    vlm_by_id = {p["id"]: p for p in vlm_pairs}
    rows = []
    for jp in jev_pairs:
        vp = vlm_by_id.get(jp["id"])
        if vp is None:
            continue
        pure, hybrid = 0.0, 0.0
        for jr, vr in zip(jp["recs"], vp["recs"]):
            pure += obs["vlm_observe_ms"] + act_ms + vr["ms"]
            hybrid += obs["jev_observe_ms"] + act_ms
            if jr["escalate"]:
                hybrid += obs["vlm_observe_ms"] + vr["ms"]  # 升级才补 SoM 截图 + VLM
            else:
                hybrid += jr["ms"]
        rows.append({
            "id": jp["id"], "steps": len(jp["recs"]),
            "vlm_pure_ms": round(pure, 1), "jev_hybrid_ms": round(hybrid, 1),
            "speedup": round(pure / hybrid, 2) if hybrid else 0.0,
            "vlm_calls_pure": len(vp["recs"]),
            "vlm_calls_hybrid": sum(1 for r in jp["recs"] if r["escalate"]),
        })
    pure_list = [r["vlm_pure_ms"] for r in rows]
    hybrid_list = [r["jev_hybrid_ms"] for r in rows]
    return {
        "episodes": len(rows), "rows": rows,
        "vlm_pure_ms_per_task": round(statistics.mean(pure_list), 1) if rows else 0.0,
        "jev_hybrid_ms_per_task": round(statistics.mean(hybrid_list), 1) if rows else 0.0,
        "speedup": round(statistics.mean(pure_list) / statistics.mean(hybrid_list), 2)
        if rows and statistics.mean(hybrid_list) else 0.0,
        "vlm_calls_per_task_pure": round(statistics.mean(
            [r["vlm_calls_pure"] for r in rows]), 2) if rows else 0.0,
        "vlm_calls_per_task_hybrid": round(statistics.mean(
            [r["vlm_calls_hybrid"] for r in rows]), 2) if rows else 0.0,
        "steps_per_task": round(statistics.mean([r["steps"] for r in rows]), 2)
        if rows else 0.0,
        "act_ms": act_ms, "observe": obs,
    }


def write_markdown(result: dict, path: Path) -> None:
    m, j, v, e = result["meta"], result.get("jev", {}), result.get("vlm"), result["e2e"]
    lines = [
        "# 北极星首测：Jev 驾驶 vs VLM 驾驶（mock replay，无头）",
        "",
        f"> 生成 {m['ts']} · ckpt `{m['ckpt']}` · 数据 `{m['mock_dir']}`（gold mock 轨迹）",
        f"> 回合 {m['episodes']}（其中 {m['vlm_episodes']} 个跑了真 VLM）· "
        f"每回合最多 {m['limit_steps']} 步 · VLM = `{m['vlm_model'] or '未启用'}`",
        "",
        "## 1. 决策时延（每步）",
        "",
        "| 驾驶者 | mean | p50 | p95 | max | n |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in (("Jev（0.6B 单次前向，GPU）", j.get("latency_ms", {})),
                    ("VLM（一次 chat 往返，含重试）", (v or {}).get("latency_ms", {}))):
        lines.append(f"| {name} | {s.get('mean', 0)} ms | {s.get('p50', 0)} ms | "
                     f"{s.get('p95', 0)} ms | {s.get('max', 0)} ms | {s.get('n', 0)} |")
    lines += [
        "",
        "## 2. 决策质量（对 oracle 标签，teacher forcing）",
        "",
        "| 驾驶者 | op_acc | joint_acc | 元素 acc | 解析失败 | 动作集外步数 |",
        "|---|---|---|---|---|---|",
        f"| Jev | {j.get('op_acc')} | {j.get('joint_acc')} | {j.get('elem_acc')} | "
        f"{j.get('errors')} 推理异常 | {j.get('steps_out_of_action_set')} |",
    ]
    if v:
        lines.append(f"| VLM | {v.get('op_acc')} | {v.get('joint_acc')} | "
                     f"{v.get('elem_acc')} | {v.get('parse_fail')} | "
                     f"{v.get('steps') - v.get('steps_in_action_set')} |")
    lines += [
        "",
        "## 3. 快路径覆盖率与升级（serve 侧校准 conf ≥ τ）",
        "",
        f"- 覆盖率 **{j.get('coverage', 0):.3f}** · 升级率 **{j.get('escalation_rate', 0):.3f}**"
        f"（τ = {m.get('tau')}，conf 为等张校准后的 P(正确)；校准文件里的覆盖率是在 "
        "10% 留出切片上算的，切片间波动可达 10 个点，见 `offline_eval.json`）",
        f"- 认输命中（oracle 也是 blocked）: {j.get('blocked_hit')} · "
        f"误认输（提前放弃）: {j.get('blocked_false')}",
        "",
        "## 4. 端到端推算（每任务）",
        "",
        f"观测成本（本机实测）：快路径 UIA 枚举 {e['observe']['jev_observe_ms']} ms · "
        f"慢路径 截图+SoM+编码 {e['observe']['vlm_observe_ms']} ms · "
        f"执行常量 {e['act_ms']} ms/步（mock UIA Invoke，两驾驶者同值可抵消）",
        "",
        "| 驾驶方式 | 每任务耗时 | VLM 调用/任务 | 相对提速 |",
        "|---|---|---|---|",
        f"| 纯 VLM（现状对照组） | {e['vlm_pure_ms_per_task']} ms | "
        f"{e['vlm_calls_per_task_pure']} | 1.00× |",
        f"| Jev 快路径 + 升级（本方案） | {e['jev_hybrid_ms_per_task']} ms | "
        f"{e['vlm_calls_per_task_hybrid']} | **{e['speedup']}×** |",
        "",
        f"（每任务平均 {e['steps_per_task']} 步；逐任务明细见 JSON）",
        "",
        "## 5. 怎么读这张表（诚实边界）",
        "",
        f"- **时延样本不对称**：Jev 是 {j.get('latency_ms', {}).get('n')} 步全量，"
        f"VLM 只有 {(v or {}).get('latency_ms', {}).get('n')} 步（真 API，按预算抽样）；"
        "VLM 的准确率数字噪声大，时延数字（p50 6s 量级）稳定。",
        "- **观测成本才是每步大头**：本机 UIA 枚举 ~0.7s，Jev 决策只有 ~0.05s —— "
        "接入后瓶颈从「决策」转到「观测」，v1 的优化重点应是增量枚举/元素表缓存。",
        "- **升级步要付两份观测**：先 UIA 枚举（快路径）再补 SoM 截图，所以升级率越高，"
        "混合模式的观测开销越接近纯 VLM；把升级率从 36% 压到 20% 的收益大于把决策再压快 10ms。",
        "- **图是合成底图 + 真实 SoM 框**：VLM 侧吃亏（无真实控件渲染），Jev 侧无图不受影响。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/jev-Qwen3-0.6B-e3")
    p.add_argument("--data", default=str(_default_home() / "traces_mock"))
    p.add_argument("--episodes", type=int, default=40, help="Jev 侧回合数（本地推理，几乎免费）")
    p.add_argument("--vlm-episodes", type=int, default=3, help="跑真 VLM 的回合数（花 API）")
    p.add_argument("--limit-steps", type=int, default=5, help="每回合最多重放多少步")
    p.add_argument("--act-ms", type=float, default=180.0, help="每步执行常数（mock UIA Invoke）")
    p.add_argument("--skip-vlm", action="store_true", help="只测 Jev（离线，零 API）")
    p.add_argument("--max-size", type=int, default=896, help="VLM 截图最长边（同线上 vlm_max_size）")
    p.add_argument("--out", default="assets/bench")
    args = p.parse_args()

    mock_dir = Path(args.data).expanduser()
    episodes = load_episodes(mock_dir, args.episodes, args.limit_steps)
    if not episodes:
        raise SystemExit(f"没有可用回合：{mock_dir}（需要 oracle 全成功的轨迹）")
    print(f"回合 {len(episodes)} × ≤{args.limit_steps} 步，数据 {mock_dir}", flush=True)

    tools = ScreenUse(dry_run=True)  # 只用于实测观测开销，不执行任何动作
    obs = measure_observe(tools)
    print(f"观测成本：Jev {obs['jev_observe_ms']}ms / VLM {obs['vlm_observe_ms']}ms", flush=True)

    from training.serve import JevDecision

    t0 = time.perf_counter()
    jev = JevDecision(args.ckpt)
    print(f"Jev 加载 {time.perf_counter() - t0:.1f}s（{jev.device}，τ={jev.tau}）", flush=True)

    t0 = time.perf_counter()
    jev_pairs = []
    for i, ep in enumerate(episodes, 1):
        jev_pairs.append(drive_jev(jev, ep))
        if i % 10 == 0:
            print(f"  Jev {i}/{len(episodes)} 回合 "
                  f"({(time.perf_counter() - t0) / i * 1000:.0f} ms/回合)", flush=True)

    vlm_pairs: list[dict] = []
    settings = get_settings()
    vlm_model = ""
    if not args.skip_vlm:
        if not settings.vlm_available:
            print("VLM 不可用（未配置/无凭据）→ 跳过对照组", flush=True)
        else:
            from screen_use.providers.openai_compat import OpenAICompatProvider

            provider = OpenAICompatProvider(settings)
            vlm_model = settings.effective_model
            subset = episodes[:args.vlm_episodes]
            for i, ep in enumerate(subset, 1):
                vlm_pairs.append(drive_vlm(provider, ep, args.max_size))
                r = vlm_pairs[-1]["recs"]
                print(f"  VLM {i}/{len(subset)} 回合：{len(r)} 步，"
                      f"{sum(x['ms'] for x in r) / 1000:.1f}s，"
                      f"解析失败 {sum(1 for x in r if not x['parsed'])}", flush=True)

    result = {
        "meta": {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ckpt": args.ckpt, "mock_dir": str(mock_dir),
            "episodes": len(episodes), "vlm_episodes": len(vlm_pairs),
            "limit_steps": args.limit_steps, "vlm_model": vlm_model,
            "tau": jev.tau, "calibrated": bool(jev._iso_x),
        },
        "jev": summarize_jev(jev_pairs),
        "vlm": summarize_vlm(vlm_pairs) if vlm_pairs else None,
        "e2e": e2e(jev_pairs, vlm_pairs, obs, args.act_ms, args.limit_steps),
        "episodes": {"jev": jev_pairs, "vlm": vlm_pairs},
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    json_path = out_dir / f"jev_vs_vlm_{stamp}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    md_path = out_dir / f"jev_vs_vlm_{stamp}.md"
    write_markdown(result, md_path)
    print(f"\nJev: {json.dumps({k: v for k, v in result['jev'].items() if k != 'latency_ms'}, ensure_ascii=False)}")
    print(f"Jev 时延: {result['jev']['latency_ms']}")
    if result["vlm"]:
        print(f"VLM 时延: {result['vlm']['latency_ms']}")
    print(f"端到端: 纯 VLM {result['e2e']['vlm_pure_ms_per_task']} ms/任务 → "
          f"Jev 混合 {result['e2e']['jev_hybrid_ms_per_task']} ms/任务 "
          f"（{result['e2e']['speedup']}×）")
    print(f"→ {json_path}\n→ {md_path}")


if __name__ == "__main__":
    main()
