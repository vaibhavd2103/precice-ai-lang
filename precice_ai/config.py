import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - only used in a minimal install
    def load_dotenv(path: Path, override: bool = False) -> bool:
        """Load simple KEY=VALUE entries when python-dotenv is unavailable."""
        if not path.exists():
            return False
        loaded = False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and (override or key not in os.environ):
                os.environ[key] = value
                loaded = True
        return loaded


# Load the repository .env for every entry point, including direct
# `uvicorn precice_ai.server:app` startup. The CLI also loads dotenv, but
# configuration must not depend on which launcher was used.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_FILE, override=False)

DEFAULT_MODEL = "openai/gpt-4o-mini"

LLM_PROVIDER: str = os.environ.get("PRECICE_AI_PROVIDER", os.environ.get("LLM_PROVIDER", "openrouter"))
LLM_API_KEY: str = os.environ.get("PRECICE_AI_API_KEY", os.environ.get("LLM_API_KEY", os.environ.get("OPENROUTER_API_KEY", "")))
MODEL: str = os.environ.get("PRECICE_AI_MODEL", os.environ.get("LLM_MODEL", DEFAULT_MODEL))
LLM_BASE_URL: str = os.environ.get(
    "PRECICE_AI_BASE_URL",
    os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1" if LLM_PROVIDER == "openrouter" else ""),
)
HOST: str = os.environ.get("PRECICE_AI_HOST", "127.0.0.1")
PORT: int = int(os.environ.get("PRECICE_AI_PORT", "7860"))
EMBED_MODEL: str = "openai/text-embedding-3-small"
CHROMA_COLLECTION: str = "precice_docs"
INGEST_INTERVAL_HOURS: int = 1
MAX_RETRIEVAL_CHUNKS: int = 6
MAX_SESSION_MESSAGES: int = 20


def llm_is_configured() -> bool:
    return bool(LLM_API_KEY and MODEL)


def public_runtime_config() -> dict:
    return {
        "provider": LLM_PROVIDER,
        "model": MODEL,
        "base_url": LLM_BASE_URL,
        "llm_configured": llm_is_configured(),
        "env_file": str(ENV_FILE),
        "env_file_exists": ENV_FILE.exists(),
        "log_file": os.environ.get("PRECICE_AI_LOG_FILE", str(ENV_FILE.parent / "logs" / "agent.jsonl")),
        "host": HOST,
        "port": PORT,
    }
