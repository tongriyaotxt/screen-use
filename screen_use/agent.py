"""视觉行为循环：observe → think → act → verify。

给定自然语言任务，循环执行：
  1. observe: 截图 + UIA 元素枚举（SoM 标注）
  2. think:   VLM 根据画面和任务决定下一步动作（JSON）
  3. act:     执行动作
  4. verify:  下一轮截图即为验证；VLM 判断任务是否完成

此功能需要 VLM（本地 Ollama 或云端均可）。

**Jev 快路径（desktop-jev 阶段 4，cascade protocol 中环）**：
构造时传 `jev`（`training/serve.py::JevDecision`）即开启三级决策路由——
每步先问 Jev（单次前向 ~0.1s，只要元素表不要截图）；Jev 高置信 → 直接执行；
低置信/不可执行 → 带着 Jev 的判断升级给 VLM 慢路径（§4 带地图退）；
Jev 输出 BLOCKED（认输）→ 不再烧 VLM，直接交宿主接管。
`jev=None` 时行为与接入前完全一致（纯 VLM 循环）。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from screen_use import jev_bridge
from screen_use.memory import ExperienceStore, TraceRecorder
from screen_use.reflect import (
    StuckSignal,
    build_reflect_prompt,
    classify_stuck,
    elements_fingerprint,
)

if TYPE_CHECKING:
    from screen_use.tools import ScreenUse

_LOOP_PROMPT = """你是一个桌面操作 Agent。根据截图和任务，决定下一步动作。

任务：{goal}

当前前台窗口：{foreground}
{experience}
{reflection}
截图上用彩色编号框标出了可交互元素。候选元素（编号: [类型] 名称）：
{legend}

已执行的历史动作：
{history}

可用动作（只输出一个 JSON，不要输出其他内容）：
- {{"thought": "分析当前状态", "action": "click_id", "id": 元素编号}}  点击某个编号元素
- {{"thought": "...", "action": "click", "x": 横坐标, "y": 纵坐标}}  点击坐标（元素表中没有时）
- {{"thought": "...", "action": "type", "text": "文本"}}
- {{"thought": "...", "action": "hotkey", "keys": "ctrl+s"}}
- {{"thought": "...", "action": "press", "key": "enter"}}
- {{"thought": "...", "action": "scroll", "clicks": -3}}
- {{"thought": "...", "action": "wait", "seconds": 1}}
- {{"thought": "...", "action": "done", "result": "任务完成的总结"}}

确定的连续操作（如 点输入框→输入→回车）可用 batch 一次提交多个动作，减少往返：
- {{"thought": "...", "batch": [{{"action": "click_id", "id": 3}}, {{"action": "type", "text": "你好"}}, {{"action": "press", "key": "enter"}}]}}

注意：截图尺寸 {width}x{height}，点击坐标需换算回真实屏幕（乘以 {scale:.1f}）。
关键规则：
0. 不要输出思考过程。直接、只输出 JSON。
1. 每步只做一个动作；确定的连续操作才用 batch（done 必须单独输出，不能放进 batch）。
2. 输入文本前，必须先点击目标窗口的输入区域获得焦点（检查当前前台窗口是否是目标），再执行 type。禁止假设焦点位置。
3. 输出 done 前，必须确认截图中能直接看到任务结果（例如记事本里真的出现了输入的文字）。禁止凭历史动作推断完成。
4. 切换窗口：直接点击候选列表中属于目标窗口的元素（点击会激活该窗口）。禁止用 alt+tab 盲目轮切。
5. 涉及键盘输入的连续操作（如 点击激活输入区→输入文本→回车、计算器连续按键）必须用 batch 一次提交；焦点可能被其他窗口抢占，分轮执行容易落空。"""

_ACTIONS = ("click_id", "click", "type", "hotkey", "press", "scroll", "wait", "done")


def _new_metrics() -> dict:
    """北极星指标原料：每一步走哪一环、各环耗时、省掉多少次截图。"""
    return {
        "wall_ms": 0.0, "steps": 0,
        "jev_steps": 0, "vlm_steps": 0, "jev_escalations": 0, "jev_errors": 0,
        "vlm_calls": 0, "screenshot_calls": 0,
        "jev_ms": 0.0, "vlm_ms": 0.0, "observe_ms": 0.0, "act_ms": 0.0,
        "ring": "none",
    }


@dataclass
class Step:
    index: int
    thought: str
    action: str
    args: dict[str, Any]
    ok: bool
    detail: str = ""
    ring: str = "vlm"  # 决策来源：jev 快路径 / vlm 慢路径


@dataclass
class TaskResult:
    goal: str
    done: bool
    result: str
    steps: list[Step] = field(default_factory=list)
    escalated: bool = False          # 由 Jev 主动认输（BLOCKED）而非步数耗尽
    escalation_reason: str = ""      # 升级原因分类（cascade protocol §4）
    metrics: dict = field(default_factory=_new_metrics)

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "done": self.done,
            "result": self.result,
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason,
            "metrics": dict(self.metrics),
            "steps": [
                {"index": s.index, "thought": s.thought, "action": s.action,
                 "args": s.args, "ok": s.ok, "detail": s.detail, "ring": s.ring}
                for s in self.steps
            ],
        }


def parse_action(raw: str) -> dict:
    """解析 VLM 的动作输出：直接 JSON → 逐个扫描平衡括号对象，取第一个含 action 的。"""
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
        for candidate in _iter_json_objects(raw):
            if isinstance(candidate, dict) and "action" in candidate:
                data = candidate
                break
        if data is None:
            raise ValueError(f"无法解析动作输出: {raw[:200]}")
    if data.get("action") not in _ACTIONS:
        raise ValueError(f"未知动作: {data.get('action')}")
    return data


def parse_actions(raw: str) -> list[dict]:
    """解析 VLM 输出为动作列表：支持单动作 JSON 或 {"batch": [动作, ...]}（向后兼容）。"""
    raw = raw.strip()
    candidates: list = []
    try:
        candidates.append(json.loads(raw))
    except json.JSONDecodeError:
        candidates.extend(_iter_json_objects(raw))
    for data in candidates:
        if not isinstance(data, dict):
            continue
        batch = data.get("batch")
        if isinstance(batch, list):
            actions = [
                a for a in batch
                if isinstance(a, dict) and a.get("action") in _ACTIONS
            ]
            if actions:
                if data.get("thought"):
                    actions[0].setdefault("thought", data["thought"])
                return actions
        if data.get("action") in _ACTIONS:
            return [data]
    raise ValueError(f"无法解析动作输出: {raw[:200]}")


def _iter_json_objects(text: str):
    """从文本中逐个提取平衡括号的 JSON 对象（容忍多个对象和前后废话）。"""
    start = 0
    while True:
        start = text.find("{", start)
        if start < 0:
            return
        depth = 0
        in_str = False
        escape = False
        end = -1
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_str = not in_str
            elif in_str:
                continue
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            return
        try:
            yield json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        start = end


class VisualLoop:
    """视觉行为循环执行器。通过 ScreenUse 实例使用：tools.run_task(goal)。

    Args:
        tools: ScreenUse 门面
        max_steps: 最大步数
        step_delay: 每步之间的固定等待
        memory: 经验存储（默认 ~/.screen_use）
        jev: 可选 Jev 快路径引擎（`.decide(goal, foreground, elements, history) -> dict`）
    """

    def __init__(self, tools: ScreenUse, max_steps: int = 40, step_delay: float = 0.5,
                 memory: ExperienceStore | None = None, jev=None):
        self.tools = tools
        self.max_steps = max_steps
        self.step_delay = step_delay
        self.memory = memory or ExperienceStore()
        self.jev = jev
        self._metrics: dict | None = None  # 当前 run 的指标桶（_observe 记截图次数用）

    def run(self, goal: str, on_step=None) -> TaskResult:
        """执行任务。on_step: 可选回调 fn(Step)，用于实时观察/日志。"""
        result = TaskResult(goal=goal, done=False, result="达到最大步数仍未完成")
        metrics = result.metrics
        self._metrics = metrics  # 供 _observe 记截图次数（run 结束清空）
        history: list[str] = []        # 中文历史（给 VLM 看）
        jev_history: list[str] = []    # 训练同构历史（给 Jev 看，格式与 SFT 一致）
        reflection = ""   # 内省结论，注入下一轮 prompt
        prev_fingerprint = ""  # 上一轮截图指纹，用于检测“动作无效”
        prev_title = ""        # 上一轮前台窗口标题，用于检测意外弹窗

        # 元学习：检索相似历史任务经验
        experience = self._recall_experience(goal)

        # desktop-jev 阶段 0：结构化逐步轨迹采集（写入失败不影响任务）
        recorder: TraceRecorder | None = None
        if self.tools.settings.record_full_trace:
            try:
                recorder = TraceRecorder(self.memory.home)
                recorder.start_task(goal)
            except OSError:
                recorder = None

        t_start = time.perf_counter()
        try:
            for i in range(1, self.max_steps + 1):
                # ---- observe：Jev 快路径只要元素表（零截图/零编码），否则照旧截图 ----
                t_obs = time.perf_counter()
                shot, legend, foreground = self._observe()
                metrics["observe_ms"] += (time.perf_counter() - t_obs) * 1000
                hist_text = "\n".join(history) if history else "(无)"

                # ---- think：Jev 快路径优先，失效/低置信才落 VLM ----
                decisions: list[dict] | None = None
                ring = "vlm"
                jev_hint = ""
                if self.jev is not None:
                    jev_decision = self._jev_decide(goal, foreground, shot, jev_history,
                                                    metrics)
                    if jev_decision is not None and jev_decision["action"] == "BLOCKED":
                        # 认输：不烧 VLM 一轮，直接按 §4 交宿主（带观测快照）
                        step = Step(index=i, thought=f"jev:{jev_decision['confidence']:.2f}",
                                    action="blocked", args={"reason": "Jev 判定无法完成"},
                                    ok=True, detail="Jev 认输（BLOCKED）", ring="jev")
                        result.steps.append(step)
                        metrics["jev_steps"] += 1
                        self._trace(recorder, step, shot)
                        if on_step:
                            on_step(step)
                        result.escalated = True
                        result.escalation_reason = "blocked_action"
                        result.result = "Jev 判定目标不可达（BLOCKED）"
                        self._record(result)
                        return result
                    decision = (jev_bridge.to_action(jev_decision)
                                if jev_decision is not None else None)
                    if decision is not None and not jev_decision.get("escalate"):
                        decisions, ring = [decision], "jev"
                        metrics["jev_steps"] += 1
                    else:
                        metrics["jev_escalations"] += 1
                        jev_hint = self._jev_hint(jev_decision)

                if decisions is None:
                    # 慢路径：Jev 升级或未接入 → 补一张 SoM 截图走 VLM
                    if shot.get("image_base64") is None:
                        t_obs = time.perf_counter()
                        shot, legend, foreground = self._observe(full=True)
                        metrics["observe_ms"] += (time.perf_counter() - t_obs) * 1000
                    prompt = _LOOP_PROMPT.format(
                        goal=goal, legend=legend, history=hist_text, foreground=foreground,
                        experience=experience, reflection=reflection,
                        width=shot["width"], height=shot["height"], scale=shot["scale"],
                    ) + jev_hint
                    reflection = ""  # 反思只生效一轮
                    t_vlm = time.perf_counter()
                    try:
                        decisions = self._think(shot["image_base64"], prompt)
                    except RuntimeError as e:
                        metrics["vlm_ms"] += (time.perf_counter() - t_vlm) * 1000
                        metrics["vlm_calls"] += 1
                        metrics["vlm_steps"] += 1  # 这一轮确实走了 VLM（只是没解析出动作）
                        step = Step(index=i, thought="", action="think_error",
                                    args={}, ok=False, detail=str(e))
                        result.steps.append(step)
                        history.append(f"第{i}步: 决策失败 -> {e}")
                        self._trace(recorder, step, shot)
                        if on_step:
                            on_step(step)
                        time.sleep(self.step_delay)
                        continue
                    metrics["vlm_ms"] += (time.perf_counter() - t_vlm) * 1000
                    metrics["vlm_calls"] += 1
                    metrics["vlm_steps"] += 1

                # ---- act：依次执行本轮决策（batch 时一轮可含多个动作）----
                for decision in decisions:
                    step = Step(
                        index=i,
                        thought=decision.get("thought", ""),
                        action=decision["action"],
                        args={k: v for k, v in decision.items()
                              if k not in ("thought", "action")},
                        ok=True,
                        ring=ring,
                    )

                    if step.action == "done":
                        result.done = True
                        result.result = str(decision.get("result", "完成"))
                        result.steps.append(step)
                        metrics["steps"] = len(result.steps)
                        self._trace(recorder, step, shot)
                        if on_step:
                            on_step(step)
                        self._record(result)
                        return result

                    t_act = time.perf_counter()
                    try:
                        detail = self._act(decision, shot.get("scale", 1.0))
                        step.detail = detail
                    except Exception as e:  # 动作失败不中断，让 VLM 下一轮看到并纠偏
                        step.ok = False
                        step.detail = f"动作失败: {e}"
                    metrics["act_ms"] += (time.perf_counter() - t_act) * 1000

                    result.steps.append(step)
                    metrics["steps"] = len(result.steps)
                    history.append(
                        f"第{i}步: {step.action} {step.args} -> {'成功' if step.ok else step.detail}"
                    )
                    jev_history.append(
                        jev_bridge.history_entry(step.action, step.args, step.ok)
                    )
                    self._trace(recorder, step, shot)
                    if on_step:
                        on_step(step)
                    if not step.ok:
                        break  # batch 中某步失败即中止本轮剩余动作，等下轮观察纠偏

                # 内省：困难分类检测 → 针对性提示词引导反思
                # 指纹取元素列表（不含每轮重排的 id），不要用带 SoM 标注的截图（标注会误变）
                fingerprint = elements_fingerprint(shot["elements"])
                title = shot.get("foreground_title") or ""
                signal = StuckSignal(
                    steps=result.steps,
                    screen_changed=fingerprint != prev_fingerprint,
                    elements_count=len(shot["elements"]),
                    window_title_changed=bool(prev_title and title and title != prev_title),
                )
                prev_fingerprint, prev_title = fingerprint, title
                stuck_type = classify_stuck(signal)
                if stuck_type and i < self.max_steps:
                    # 复用本轮截图（带 SoM 标注），不再重复截图；快路径无图时补一张
                    if shot.get("image_base64") is None:
                        shot, legend, foreground = self._observe(full=True)
                    reflection = self._reflect(stuck_type, goal, history,
                                               shot["image_base64"])
                    history.append(f"[反思·{stuck_type.value}] {reflection}")

                time.sleep(self.step_delay)
        finally:
            metrics["wall_ms"] = (time.perf_counter() - t_start) * 1000
            metrics["ring"] = ("hybrid" if metrics["jev_steps"] and metrics["vlm_steps"]
                               else "jev" if metrics["jev_steps"]
                               else "vlm" if metrics["vlm_steps"] else "none")
            self._metrics = None

        self._record(result)
        return result

    # ================= 观测 =================

    def _observe(self, full: bool = False) -> tuple[dict, str, str]:
        """观测 → (shot, legend, foreground)。

        full=False 且接了 Jev 时只枚举 UIA 元素表（零截图、零 JPEG 编码，
        快路径每步省的就是这块）；full=True 或纯 VLM 模式走原 screenshot(annotate)。
        """
        if self.jev is not None and not full:
            data = self.tools.list_ui_elements()
            shot = {
                "image_base64": None, "scale": 1.0, "width": 0, "height": 0,
                "foreground_title": data["foreground_title"],
                "elements": data["elements"],
            }
        else:
            shot = self.tools.screenshot(
                annotate=True, max_size=self.tools.settings.vlm_max_size
            )
            m = getattr(self, "_metrics", None)
            if m is not None:
                m["screenshot_calls"] += 1  # 北极星指标：快路径省下的截图次数
        return shot, _legend(shot["elements"]), (shot.get("foreground_title") or "(未知)")

    # ================= Jev 快路径 =================

    def _jev_decide(self, goal: str, foreground: str, shot: dict,
                    jev_history: list[str], metrics: dict) -> dict | None:
        """问一次 Jev（异常只降级不中断）。返回 None = 本步直接走 VLM。"""
        t0 = time.perf_counter()
        try:
            decision = self.jev.decide(goal, foreground, shot["elements"],
                                       list(jev_history[-4:]))
        except Exception as e:  # noqa: BLE001 - 加速组件不能把主流程拖挂
            metrics["jev_errors"] += 1
            metrics["jev_ms"] += (time.perf_counter() - t0) * 1000
            shot["_jev_error"] = str(e)
            return None
        metrics["jev_ms"] += float(decision.get("latency_ms")
                                   or (time.perf_counter() - t0) * 1000)
        return decision

    @staticmethod
    def _jev_hint(decision: dict | None) -> str:
        """升级时的「带地图退」：把 Jev 的判断和置信度告诉 VLM，但不让它盲从。"""
        if decision is None:
            return ""
        op = decision.get("action", "?")
        conf = float(decision.get("confidence") or 0.0)
        bits = [f"快速决策器（小模型）倾向 {op}"]
        if decision.get("element_id") is not None:
            bits.append(f"元素 #{decision['element_id']}")
        if decision.get("text"):
            bits.append(f"文本 {decision['text']!r}")
        bits.append(f"置信度 {conf:.2f} 低于阈值")
        return ("\n【快路径提示】" + "，".join(bits)
                + "。请独立判断，不要盲从；若你也认为无法完成，请输出 done 说明原因。")

    def _trace(self, recorder: TraceRecorder | None, step: Step, shot: dict) -> None:
        """desktop-jev 阶段 0：落盘一步「元素表 + 决策 + 结果」（写入失败不影响任务）。"""
        if recorder is None:
            return
        try:
            recorder.record_step(
                step.index, shot.get("foreground_title") or "",
                shot.get("elements", []), step.action, step.args, step.ok, step.detail,
                source="jev" if step.ring == "jev" else "vlm_trace",
            )
        except OSError:
            pass

    def _record(self, result: TaskResult) -> None:
        """元学习：任务结束记录轨迹（写入失败不影响任务结果）。"""
        try:
            self.memory.record_trace(result.goal, result.to_dict())
        except OSError:
            pass

    # ================= 元学习 =================

    def _recall_experience(self, goal: str) -> str:
        """检索相似历史任务，生成经验提示文本。"""
        try:
            similar = self.memory.recall(goal)
        except OSError:
            return ""
        if not similar:
            return ""
        lines = ["相似任务的历史经验（可参考）："]
        for e in similar:
            status = "成功" if e.get("done") else "失败"
            actions = " → ".join(
                f"{s['action']}({json.dumps(s['args'], ensure_ascii=False)})"
                for s in e.get("steps", [])[:8]
            )
            lines.append(f"- 任务 {e['goal']!r}（{status}）: {actions}")
        return "\n".join(lines) + "\n"

    # ================= 内省 =================

    def _reflect(self, stuck_type, goal: str, history: list[str], image_base64: str) -> str:
        """内省：按困难类型生成针对性提示词，让 VLM 分析原因并调整策略。

        复用本轮 observe 的截图（带 SoM 标注），不重复截图。
        """
        try:
            prompt = build_reflect_prompt(stuck_type, goal, history)
            raw = self.tools.provider.ask_about_screen(image_base64, prompt)
            data = next(iter(_iter_json_objects(raw)), {})
            return f"原因: {data.get('analysis', '')} | 策略: {data.get('strategy', '')}"
        except Exception as e:
            return f"(反思失败: {e})"

    def _think(self, image_base64: str, prompt: str) -> list[dict]:
        provider = self.tools.provider  # VLM 未配置时在此抛错
        last_error: Exception | None = None
        current_prompt = prompt
        for _ in range(2):  # 解析失败重试 1 次，并把错误原因反馈给模型（纠正式重试，非盲重试）
            try:
                raw = provider.ask_about_screen(image_base64, current_prompt)
                return parse_actions(raw)
            except ValueError as e:
                last_error = e
                current_prompt = prompt + (
                    f"\n\n【纠错】你上次的输出无法解析为动作 JSON：{e}\n"
                    "请只输出一个 JSON 对象，不要输出任何其他内容。"
                )
        raise RuntimeError(f"VLM 动作决策失败: {last_error}")

    def _act(self, decision: dict, scale: float) -> str:
        t = self.tools
        action = decision["action"]
        if action == "click_id":
            r = t.click_element_id(int(decision["id"]))
            return f"点击元素 #{decision['id']}"
        if action == "double_click":
            if decision.get("id") is not None:
                t.click_element_id(int(decision["id"]))
                return f"双击元素 #{decision['id']}"
            x, y = int(decision["x"] * scale), int(decision["y"] * scale)
            t.double_click(x, y)
            return f"双击 ({x}, {y})"
        if action == "right_click":
            if decision.get("id") is not None:
                t.click_element_id(int(decision["id"]))
                return f"右键元素 #{decision['id']}"
            x, y = int(decision["x"] * scale), int(decision["y"] * scale)
            t.right_click(x, y)
            return f"右键 ({x}, {y})"
        if action == "click":
            # 截图坐标 → 真实屏幕坐标
            x, y = int(decision["x"] * scale), int(decision["y"] * scale)
            t.click(x, y)
            return f"点击 ({x}, {y})"
        if action == "type":
            t.type_text(str(decision["text"]))
            return f"输入 {decision['text']!r}"
        if action == "hotkey":
            t.hotkey(*str(decision["keys"]).split("+"))
            return f"快捷键 {decision['keys']}"
        if action == "press":
            t.press(str(decision["key"]))
            return f"按键 {decision['key']}"
        if action == "scroll":
            t.scroll(int(decision["clicks"]))
            return f"滚动 {decision['clicks']}"
        if action == "wait":
            time.sleep(min(float(decision.get("seconds", 1)), 10))
            return "等待"
        raise ValueError(f"不支持的动作: {action}")


def _legend(elements: list[dict]) -> str:
    return "\n".join(
        f"{el['id']}: [{el['window_title']}] [{el['control_type']}] {el['name'] or '(无名)'}"
        for el in elements
    ) or "(无可交互元素)"
