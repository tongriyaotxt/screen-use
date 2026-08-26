"""配置管理：读取 .env，VLM 为可选配置。"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# 内置 Provider 预设
PROVIDER_PRESETS = {
    "openai": ("https://api.openai.com/v1", "gpt-4o"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-vl-max"),
    "ollama": ("http://localhost:11434/v1", "qwen2.5vl:7b"),
    # Kimi Code 订阅（OAuth 走 ~/.kimi/credentials，access token 动态读取）
    "kimi-code": ("https://api.kimi.com/coding/v1", "kimi-for-coding"),
}

# Kimi CLI 的 OAuth 凭据文件（kimi CLI 会自动刷新 access_token）
_KIMI_CREDENTIALS = Path.home() / ".kimi" / "credentials" / "kimi-code.json"


def _kimi_access_token() -> str:
    """读取 Kimi CLI 的当前 OAuth access_token（每次调用实时读，天然跟随刷新）。"""
    try:
        return json.loads(_KIMI_CREDENTIALS.read_text(encoding="utf-8"))["access_token"]
    except Exception:
        return ""


class Settings(BaseSettings):
    """全局配置。VLM 相关字段均为可选 —— 未配置时策略链自动降级。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # VLM（可选）
    vision_provider: str = "ollama"
    vision_base_url: str = ""
    vision_api_key: str = ""
    vision_model: str = ""

    # 截图
    screenshot_max_size: int = 1280
    screenshot_jpeg_quality: int = 80
    screenshot_max_bytes: int = 90 * 1024  # JPEG 字节上限，超限自动降 quality 重编码（防 MCP 输出被宿主截断）

    # VLM 专用截图尺寸：VLM 内部会把图切成 token 网格，1280px ≈ 上千 image token。
    # 发给 VLM 的图单独用更小的尺寸可显著降低 TTFT；click_id 路径按元素 id 定位，不依赖分辨率。
    vlm_max_size: int = 896

    @property
    def vlm_available(self) -> bool:
        """VLM 是否可用：本地 Ollama 不需要 API key，云端需要。"""
        if not self.effective_model:
            return False
        if self.vision_provider == "ollama":
            return True
        if self.vision_provider == "kimi-code":
            return bool(_kimi_access_token())
        return bool(self.vision_api_key)

    @property
    def effective_base_url(self) -> str:
        if self.vision_base_url:
            return self.vision_base_url
        preset = PROVIDER_PRESETS.get(self.vision_provider)
        return preset[0] if preset else ""

    @property
    def effective_model(self) -> str:
        if self.vision_model:
            return self.vision_model
        preset = PROVIDER_PRESETS.get(self.vision_provider)
        return preset[1] if preset else ""

    @property
    def effective_api_key(self) -> str:
        # kimi-code：动态读 OAuth access_token（15 分钟轮换，不能写死）
        if self.vision_provider == "kimi-code":
            return _kimi_access_token()
        # Ollama 的 OpenAI 兼容端点要求传任意非空 key
        return self.vision_api_key or "ollama"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
