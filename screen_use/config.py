"""配置管理：读取 .env，VLM 为可选配置。"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


# 内置 Provider 预设
PROVIDER_PRESETS = {
    "openai": ("https://api.openai.com/v1", "gpt-4o"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-vl-max"),
    "ollama": ("http://localhost:11434/v1", "qwen2.5vl:7b"),
}


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

    @property
    def vlm_available(self) -> bool:
        """VLM 是否可用：本地 Ollama 不需要 API key，云端需要。"""
        if not self.effective_model:
            return False
        if self.vision_provider == "ollama":
            return True
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
        # Ollama 的 OpenAI 兼容端点要求传任意非空 key
        return self.vision_api_key or "ollama"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
