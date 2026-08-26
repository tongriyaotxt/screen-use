"""OpenAI 兼容 VLM Provider：一份实现覆盖 GPT-4o / Qwen-VL / Ollama。

Ollama 自带 OpenAI 兼容端点（http://localhost:11434/v1），
DashScope 提供兼容模式，因此统一走 chat.completions 协议。
"""

from __future__ import annotations

import json
import re

from openai import APIConnectionError, APIStatusError, OpenAI

from screen_use.config import Settings
from screen_use.providers.base import VisionProvider

_PICK_PROMPT = """你是一个桌面 UI 定位助手。截图上用彩色编号框标出了可交互的 UI 元素。

候选元素列表（编号: [类型] 名称）：
{legend}

用户目标：{goal}

请判断哪个编号最符合用户目标。
只输出 JSON，不要输出任何其他内容：
{{"id": 编号, "reason": "一句话理由"}}
如果没有任何元素符合目标，输出：{{"id": -1, "reason": "原因"}}"""

_SCREEN_QA_PROMPT = """你是一个屏幕阅读助手。请根据截图回答用户的问题，简洁准确。
如果问题涉及具体文字内容，请尽量原文引用。

用户问题：{question}"""


def extract_json(text: str) -> dict:
    """从模型输出中提取 JSON：先直接解析，失败则正则兜底。"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 兜底：提取第一个 {...} 块（容忍 markdown 代码围栏和前后废话）
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"无法从模型输出中提取 JSON: {text[:200]}")


class OpenAICompatProvider(VisionProvider):
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = OpenAI(
            base_url=settings.effective_base_url,
            api_key=settings.effective_api_key,
            timeout=60,  # 防止慢模型/网络问题导致无限挂起
        )

    def _chat_with_image(self, prompt: str, image_base64: str) -> str:
        # 每次请求前刷新 key（kimi-code 的 OAuth token 会轮换）
        self._client.api_key = self._settings.effective_api_key
        kwargs: dict = {}
        # Ollama：关 thinking + 扩大上下文窗口（默认 4096，图片+prompt+思考容易爆）
        if self._settings.vision_provider == "ollama":
            kwargs["extra_body"] = {"think": False, "options": {"num_ctx": 16384}}
        # kimi-for-coding 仅允许 temperature=1；其他模型用 0 求稳定
        temperature = 1 if self._settings.vision_provider == "kimi-code" else 0
        try:
            resp = self._client.chat.completions.create(
                model=self._settings.effective_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=temperature,
                max_tokens=1024,  # 动作 JSON 足够；过大只会拖慢思考型模型
                **kwargs,
            )
        except APIConnectionError as e:
            raise RuntimeError(
                f"VLM 连接失败（{self._settings.effective_base_url}）："
                "请确认 Ollama 已启动或 API 配置正确"
            ) from e
        except APIStatusError as e:
            if e.status_code >= 500:
                # 服务端错误（上下文溢出/runner 异常等）按可重试处理
                raise ValueError(f"VLM 服务端错误 {e.status_code}") from e
            raise
        choice = resp.choices[0]
        content = choice.message.content or ""
        if not content.strip():
            # 思考型模型可能把输出写进 reasoning 字段（think:false 在部分 Ollama 版本不生效）
            reasoning = (
                getattr(choice.message, "reasoning", None)
                or getattr(choice.message, "reasoning_content", None)
                or ""
            )
            if reasoning.strip():
                return reasoning  # 交给上层从推理文本中提取 JSON
            raise ValueError(f"VLM 返回空内容 (finish_reason={choice.finish_reason})")
        return content

    def pick_element(self, image_base64: str, legend: str, goal: str) -> int:
        prompt = _PICK_PROMPT.format(legend=legend, goal=goal)
        last_error: Exception | None = None
        last_raw = ""
        for attempt in range(2):  # 解析失败重试 1 次
            try:
                raw = self._chat_with_image(prompt, image_base64)
                last_raw = raw
                data = extract_json(raw)
                return int(data.get("id", -1))
            except (ValueError, KeyError, TypeError) as e:
                last_error = e
        raise RuntimeError(
            f"VLM 元素定位失败（2 次尝试均无法解析）：{last_error}；"
            f"最后一次原始输出: {last_raw[:200]!r}"
        )

    def ask_about_screen(self, image_base64: str, question: str) -> str:
        prompt = _SCREEN_QA_PROMPT.format(question=question)
        return self._chat_with_image(prompt, image_base64)
