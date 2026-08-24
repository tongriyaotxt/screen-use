"""Provider 测试：JSON 解析 + VLM 响应处理（mock HTTP）。"""

from unittest.mock import MagicMock, patch

import pytest

from screen_use.config import Settings
from screen_use.providers.openai_compat import OpenAICompatProvider, extract_json


# ---- extract_json ----

def test_extract_json_plain():
    assert extract_json('{"id": 3, "reason": "ok"}') == {"id": 3, "reason": "ok"}


def test_extract_json_with_markdown_fence():
    text = '```json\n{"id": 2, "reason": "匹配"}\n```'
    assert extract_json(text)["id"] == 2


def test_extract_json_with_surrounding_text():
    text = '我认为答案是 {"id": 5} 因为它在最上面'
    assert extract_json(text)["id"] == 5


def test_extract_json_invalid_raises():
    with pytest.raises(ValueError):
        extract_json("没有 JSON 在这里")


# ---- OpenAICompatProvider ----

def _settings():
    return Settings(
        vision_provider="custom",
        vision_base_url="http://fake/v1",
        vision_api_key="fake",
        vision_model="fake-vlm",
    )


def _mock_provider(response_text: str) -> OpenAICompatProvider:
    provider = OpenAICompatProvider(_settings())
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = response_text
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = mock_resp
    return provider


def test_pick_element_success():
    provider = _mock_provider('{"id": 7, "reason": "保存按钮"}')
    assert provider.pick_element("b64img", "7: [Button] 保存", "保存按钮") == 7


def test_pick_element_not_found():
    provider = _mock_provider('{"id": -1, "reason": "没有"}')
    assert provider.pick_element("b64img", "", "不存在的按钮") == -1


def test_pick_element_retry_on_bad_json():
    provider = _mock_provider("这不是 JSON")
    # 第一次解析失败 → 重试 → 第二次还是失败 → 抛异常
    with pytest.raises(RuntimeError, match="定位失败"):
        provider.pick_element("b64img", "", "按钮")
    assert provider._client.chat.completions.create.call_count == 2


def test_ask_about_screen():
    provider = _mock_provider("弹窗提示保存失败")
    assert provider.ask_about_screen("b64img", "弹窗说了什么？") == "弹窗提示保存失败"
