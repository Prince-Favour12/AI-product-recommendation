from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """Base configuration shared across environments."""

    model_config = SettingsConfigDict(
        env_file=str(".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # Qdrant configuration
    QDRANT_HOST: str = Field(..., env="QDRANT_HOST")
    QDRANT_PORT: int = Field(..., env="QDRANT_PORT")
    COLLECTION_NAME: str = Field(..., env="COLLECTION_NAME")

    # Database configuration
    POSTGRES_USER: str = Field(..., env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(..., env="POSTGRES_PASSWORD")
    POSTGRES_DB: str = Field(..., env="POSTGRES_DB")
    POSTGRES_HOST: str = Field(..., env="POSTGRES_HOST")
    POSTGRES_PORT: int = Field(..., env="POSTGRES_PORT")


    # data path
    DATA_PATH: str = Field(..., env="DATA_PATH")
    EMBEDDING_MODEL_NAME: str = Field(..., env="EMBEDDING_MODEL_NAME")


settings = Settings()