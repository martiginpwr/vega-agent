from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://127.0.0.1:11434"
    vega_default_model: str = ""
    vega_database_path: str = "storage/vega.db"
    vega_memory_model: str = ""
    vega_embedding_model: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
