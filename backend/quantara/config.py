from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

# Only the integrated application's root config, never a prototype's backend/.env.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./runtime/quantara.db")
    runtime: Path = Path(os.getenv("RUNTIME_DIR", "runtime")).resolve()
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "qwen3.5:4b")
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "180"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "600"))
    llm_keep_alive: str = os.getenv("LLM_KEEP_ALIVE", "30m")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    demo_mode: bool = os.getenv("DEMO_MODE", "true").lower() == "true"
    task_backend: str = os.getenv("TASK_BACKEND", "local")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    cookie_secure: bool = os.getenv("COOKIE_SECURE", "false").lower() == "true"
    origins: tuple[str, ...] = tuple(
        os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3100,http://127.0.0.1:3100",
        ).split(",")
    )


settings = Settings()
# Keep downloaded research weights beside other local runtime data.
os.environ.setdefault("HF_HOME", str(settings.runtime / "huggingface"))
