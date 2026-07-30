"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field

# Azure OpenAI credentials (AZURE_OPENAI_*) genuinely identical across Agent
# 1/2/3 live in the repo-root .env — loaded here as a fallback UNDER this
# service's own .env, mirroring the exact pattern Agent 1/3 already use
# (local always wins on any key present in both).
_ROOT_ENV = str(Path(__file__).resolve().parent.parent.parent.parent / ".env")


class Settings(BaseSettings):
    """Central application settings with environment variable binding."""

    # Database
    database_url: str = Field(
        default="postgresql://mva_user:mva_password@localhost:5432/mva_profiling",
        alias="DATABASE_URL",
    )
    database_echo: bool = Field(default=False, alias="DATABASE_ECHO")

    # File Limits
    max_upload_size_mb: int = Field(default=25, alias="MAX_UPLOAD_SIZE_MB")
    max_dataset_rows: int = Field(default=200_000, alias="MAX_DATASET_ROWS")
    max_dataset_columns: int = Field(default=200, alias="MAX_DATASET_COLUMNS")
    processing_timeout_seconds: int = Field(default=120, alias="PROCESSING_TIMEOUT_SECONDS")

    # Temporary Storage
    temp_storage_dir: str = Field(default="./tmp/uploads", alias="TEMP_STORAGE_DIR")
    max_sample_values: int = Field(default=10, alias="MAX_SAMPLE_VALUES")

    # Drill-down Cubes
    min_cube_group_size: int = Field(default=5, alias="MIN_CUBE_GROUP_SIZE")

    # LLM Provider — "azure" (default, real backend) or "mock" (tests only,
    # no network calls). No more "groq"/"openai" branch: this project runs
    # on Azure OpenAI exclusively.
    llm_provider: str = Field(default="azure", alias="LLM_PROVIDER")

    # Azure OpenAI — this project runs on Azure OpenAI, not Groq. These 4
    # normally come from the shared root .env (see _ROOT_ENV above); a
    # service-local .env value always wins if both define the same key.
    azure_openai_api_key: str = Field(default="", alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment: str = Field(default="", alias="AZURE_OPENAI_DEPLOYMENT")
    # Only used by the feature-target/rule-suggestion ReAct agents'
    # AzureChatOpenAI client — LangChain's Azure integration needs an
    # explicit api-version, unlike the direct-httpx LLMProvider calls.
    azure_openai_api_version: str = Field(default="2024-10-21", alias="AZURE_OPENAI_API_VERSION")
    llm_timeout_seconds: int = Field(default=30, alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=2, alias="LLM_MAX_RETRIES")
    llm_burst_cooldown_seconds: float = Field(default=5.0, alias="LLM_BURST_COOLDOWN_SECONDS")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")

    # Application
    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")

    # Authentication
    api_keys: list[str] = Field(default_factory=list, alias="API_KEYS")
    jwt_secret: str = Field(default="", alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")

    @property
    def max_upload_size_bytes(self) -> int:
        """Maximum upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    model_config = {
        "env_file": (_ROOT_ENV, ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Factory function for settings singleton."""
    return Settings()
