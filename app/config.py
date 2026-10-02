from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = ""
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = ""
    local_base_url: str = "http://127.0.0.1:8080/v1/"
    local_api_key: SecretStr = SecretStr("local")
    local_model: str = "local-model"
    minicpm_base_url: str = "http://127.0.0.1:8081/v1/"
    minicpm_api_key: SecretStr = SecretStr("local")
    minicpm_model: str = "MiniCPM5-2B"
    request_timeout_seconds: float = Field(default=120, gt=0)

    anthropic_api_key: SecretStr = SecretStr("")
    anthropic_model: str = ""
    local_data_residency: str | None = None
    memory_path: str = ".data/memory.sqlite3"
    # Deployment facts are operator-owned, never supplied by an incoming request.
    # Keyed by provider; set exact model, region, pricing and approvals for clouds.
    model_metadata: dict[str, dict] = Field(default_factory=dict)
