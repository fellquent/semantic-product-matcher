"""Engine settings. Every field can be passed directly or set through an
environment variable / `.env` entry of the same name, e.g. SIMILARITY_THRESHOLD=0.6."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class MatcherConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Empty → the Anthropic SDK finds credentials itself (ANTHROPIC_API_KEY, `ant auth login` profile).
    anthropic_api_key: str = ""
    claude_model: str = "claude-haiku-4-5"
    embedding_model: str = "BAAI/bge-m3"

    # Stage 1: listings with cosine similarity below this never reach the LLM.
    similarity_threshold: float = 0.55

    max_retries: int = 5          # SDK retries on 429 / 5xx / overloaded / network errors
    max_tokens: int = 512
