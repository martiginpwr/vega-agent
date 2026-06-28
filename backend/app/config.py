from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MEMORY_MODEL = "qwen3.5:9b"
DEFAULT_MEMORY_GROUNDING_MODEL = "qwen3.5:9b"
DEFAULT_EMBEDDING_MODEL = "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"


class Settings(BaseSettings):
    ollama_base_url: str = "http://127.0.0.1:11434"
    vega_default_model: str = "qwen3.5:9b"
    vega_database_path: str = "storage/vega.db"
    vega_memory_model: str = DEFAULT_MEMORY_MODEL
    vega_memory_grounding_model: str = DEFAULT_MEMORY_GROUNDING_MODEL
    vega_memory_verifier_model: str = DEFAULT_MEMORY_MODEL
    vega_embedding_model: str = DEFAULT_EMBEDDING_MODEL

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
