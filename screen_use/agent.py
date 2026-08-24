"""视觉行为循环：observe → think → act → verify。

给定自然语言任务，循环执行：
  1. observe: 截图 + UIA 元素枚举（SoM 标注）
  2. think:   VLM 根据画面和任务决定下一步动作（JSON）
  3. act:     执行动作
  4. verify:  下一轮截图即为验证；VLM 判断任务是否完成

此功能需要 VLM（本地 Ollama 或云端均可）。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from screen_use.memory import ExperienceStore
from screen_use.reflect import (
    StuckSignal,
    build_reflect_prompt,
    classify_stuck,
    screen_fingerprint,
)

if TYPE_CHECKING:
    from screen_use.tools import ScreenUse

_LOOP_PROMPT = """你是一个桌面操作 Agent。根据截图和任务，决定下一步动作。

任务：{goal}
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

注意：截图尺寸 {width}x{height}，点击坐标需换算回真实屏幕（乘以 {scale:.1f}）。
每步只做一个动作。任务完成时输出 done。"""

_ACTIONS = ("click_id", "click", "type", "hotkey", "press", "scroll", "wait", "done")


@dataclass
class Step:
    index: int
    thought: str
    action: str
    args: dict[str, Any]
    ok: bool
    detail: str = ""


@dataclass
class TaskResult:
    goal: str
    done: bool
    result: str
    steps: list[Step] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "done": self.done,
            "result": self.result,
            "steps": [
                {"index": s.index, "thought": s.thought, "action": s.action,
                 "args": s.args, "ok": s.ok, "detail": s.detail}
                for s in self.steps
            ],
        }


def parse_action(raw: str) -> dict:
    """解析 VLM 的动作输出：直接 JSON → 正则兜底。"""
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"无法解析动作输出: {raw[:200]}")
        data = json.loads(match.group(0))
    if data.get("action") not in _ACTIONS:
        raise ValueError(f"未知动作: {data.get('action')}")
    return data


class VisualLoop:
    """视觉行为循环执行器。通过 ScreenUse 实例使用：tools.run_task(goal)。"""

    def __init__(self, tools: ScreenUse, max_steps: int = 20, step_delay: float = 0.5,
                 memory: ExperienceStore | None = None):
        self.tools = tools
        self.max_steps = max_steps
        self.step_delay = step_delay
        self.memory = memory or ExperienceStore()

    def run(self, goal: str, on_step=None) -> TaskResult:
        """执行任务。on_step: 可选回调 fn(Step)，用于实时观察/日志。"""
        result = TaskResult(goal=goal, done=False, result="达到最大步数仍未完成")
        history: list[str] = []
        reflection = ""   # 内省结论，注入下一轮 prompt
        prev_fingerprint = ""  # 上一轮截图指纹，用于检测“动作无效”
        prev_title = ""        # 上一轮前台窗口标题，用于检测意外弹窗

        # 元学习：检索相似历史任务经验
        experience = self._recall_experience(goal)

        for i in range(1, self.max_steps + 1):
            # observe
            shot = self.tools.screenshot(annotate=True)
            legend = "\n".join(
                f"{el['id']}: [{el['control_type']}] {el['name'] or '(无名)'}"
                for el in shot["elements"]
            ) or "(无可交互元素)"
            hist_text = "\n".join(history) if history else "(无)"

            # think
            prompt = _LOOP_PROMPT.format(
                goal=goal, legend=legend, history=hist_text,
                experience=experience, reflection=reflection,
                width=shot["width"], height=shot["height"], scale=shot["scale"],
            )
            decision = self._think(shot["image_base64"], prompt)
            reflection = ""  # 反思只生效一轮

            step = Step(
                index=i,
                thought=decision.get("thought", ""),
                action=decision["action"],
                args={k: v for k, v in decision.items() if k not in ("thought", "action")},
                ok=True,
            )

            # act
            if step.action == "done":
                result.done = True
                result.result = str(decision.get("result", "完成"))
                result.steps.append(step)
                if on_step:
                    on_step(step)
                self._record(result)
                return result

            try:
                detail = self._act(decision, shot["scale"])
                step.detail = detail
            except Exception as e:  # 动作失败不中断，让 VLM 下一轮看到并纠偏
                step.ok = False
                step.detail = f"动作失败: {e}"

            result.steps.append(step)
            history.append(
                f"第{i}步: {step.action} {step.args} -> {'成功' if step.ok else step.detail}"
            )
            if on_step:
                on_step(step)

            # 内省：困难分类检测 → 针对性提示词引导反思
            fingerprint = screen_fingerprint(shot["image_base64"])
            title = shot["elements"][0]["window_title"] if shot["elements"] else ""
            signal = StuckSignal(
                steps=result.steps,
                screen_changed=fingerprint != prev_fingerprint,
                elements_count=len(shot["elements"]),
                window_title_changed=bool(prev_title and title and title != prev_title),
            )
            prev_fingerprint, prev_title = fingerprint, title
            stuck_type = classify_stuck(signal)
            if stuck_type and i < self.max_steps:
                reflection = self._reflect(stuck_type, goal, history)
                history.append(f"[反思·{stuck_type.value}] {reflection}")

            time.sleep(self.step_delay)

        self._record(result)
        return result

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

    def _reflect(self, stuck_type, goal: str, history: list[str]) -> str:
        """内省：按困难类型生成针对性提示词，让 VLM 分析原因并调整策略。"""
        try:
            from screen_use.perception import screen as screen_mod

            _, b64, _ = screen_mod.capture_for_model(
                self.tools.settings.screenshot_max_size,
                self.tools.settings.screenshot_jpeg_quality,
            )
            prompt = build_reflect_prompt(stuck_type, goal, history)
            raw = self.tools.provider.ask_about_screen(b64, prompt)
            data = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
            return f"原因: {data.get('analysis', '')} | 策略: {data.get('strategy', '')}"
        except Exception as e:
            return f"(反思失败: {e})"

    def _think(self, image_base64: str, prompt: str) -> dict:
        provider = self.tools.provider  # VLM 未配置时在此抛错
        last_error: Exception | None = None
        for _ in range(2):  # 解析失败重试 1 次
            try:
                raw = provider.ask_about_screen(image_base64, prompt)
                return parse_action(raw)
            except ValueError as e:
                last_error = e
        raise RuntimeError(f"VLM 动作决策失败: {last_error}")

    def _act(self, decision: dict, scale: float) -> str:
        t = self.tools
        action = decision["action"]
        if action == "click_id":
            r = t.click_element_id(int(decision["id"]))
            return f"点击元素 #{decision['id']}"
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
