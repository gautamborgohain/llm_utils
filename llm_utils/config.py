from pydantic_settings import BaseSettings
import os
from typing import List

custom_gguf_models = "rr_model_v2"


class Config(BaseSettings):
    GOOGLE_CLOUD_PROJECT_ID: str = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    CUSTOM_GGUF_MODELS: List[str] = os.getenv(
        "CUSTOM_GGUF_MODELS", custom_gguf_models
    ).split(",")
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")


config = Config()
