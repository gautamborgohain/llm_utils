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
    default_rate_limiter_requests_per_second: float = os.getenv(
        "DEFAULT_RATE_LIMITER_REQUESTS_PER_SECOND", 10
    )
    default_rate_limiter_check_every_n_seconds: float = os.getenv(
        "DEFAULT_RATE_LIMITER_CHECK_EVERY_N_SECONDS", 0.5
    )
    default_rate_limiter_max_bucket_size: int = os.getenv(
        "DEFAULT_RATE_LIMITER_MAX_BUCKET_SIZE", 10
    )


config = Config()
