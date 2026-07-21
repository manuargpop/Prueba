import os
from dotenv import load_dotenv

load_dotenv()


def _get_bool_env(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"false", "0", "no", "off"}


class Settings:
    ALIBABA_API_KEY: str = os.getenv("ALIBABA_API_KEY", "")
    ALIBABA_BASE_URL: str = os.getenv("ALIBABA_BASE_URL", "y aqui el link de la apykey")
    ALIBABA_VERIFY_SSL: bool = _get_bool_env("ALIBABA_VERIFY_SSL", True)
    REQUESTS_CA_BUNDLE: str | None = os.getenv("REQUESTS_CA_BUNDLE")
    ALIBABA_MODEL: str = os.getenv("ALIBABA_MODEL", "qwen3.7-plus")


settings = Settings()