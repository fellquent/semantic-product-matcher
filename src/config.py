from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    embedding_model: str = "BAAI/bge-m3"
    claude_model: str = "claude-haiku-4-5"
    faiss_top_k: int = 50
    llm_top_k: int = 20
    embedding_dim: int = 1024
    similarity_threshold: float = 0.55

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
