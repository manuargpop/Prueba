import os
from dotenv import load_dotenv

load_dotenv()


def _get_bool_env(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"false", "0", "no", "off"}


class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_VERIFY_SSL: bool = _get_bool_env("GROQ_VERIFY_SSL", True)
    REQUESTS_CA_BUNDLE: str | None = os.getenv("REQUESTS_CA_BUNDLE")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


settings = Settings()
