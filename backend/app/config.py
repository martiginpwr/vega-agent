from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MEMORY_MODEL = "hf.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF:latest"


class Settings(BaseSettings):
    ollama_base_url: str = "http://127.0.0.1:11434"
    vega_default_model: str = ""
    vega_database_path: str = "storage/vega.db"
    vega_memory_model: str = DEFAULT_MEMORY_MODEL
    vega_memory_verifier_model: str = DEFAULT_MEMORY_MODEL
    vega_embedding_model: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
