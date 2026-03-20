from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://memory:memory@localhost:5432/memory"
    DB_POOL_MIN: int = 5
    DB_POOL_MAX: int = 20

    REDIS_URL: str = "redis://localhost:6379/0"

    EXTRACT_MODEL: str = "gpt-4.1-mini"
    ASSEMBLE_MODEL: str = "gpt-4.1-mini"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str | None = None
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_RETRIES: int = 3

    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_DEVICE: str = "cpu"

    SESSION_TIME_GAP_MINUTES: int = 30
    SESSION_MAX_MESSAGES: int = 50
    SESSION_MAX_TOKENS: int = 8192
    SESSION_MIN_MESSAGES: int = 2
    EXTRACT_SCAN_INTERVAL_SECONDS: int = 30

    PROFILE_FACT_THRESHOLD: int = 10
    PROFILE_TIME_THRESHOLD_HOURS: int = 24
    PROFILE_FORCE_ON_DECLARATION: bool = True

    DEFAULT_DECAY_RATE: float = 0.01
    DECAY_SWEEP_INTERVAL_HOURS: int = 6

    DEFAULT_TENANT_ID: str = "default"

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"

    model_config = {"env_prefix": "MEM_", "env_file": ".env"}


settings = Settings()
