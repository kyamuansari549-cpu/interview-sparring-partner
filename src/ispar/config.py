"""Settings — everything comes from env vars (see .env.example).

The defaults below are the setup that actually runs on my laptop:
cloud LLM with Groq first (openai/gpt-oss-120b) and Gemini as fallback,
whisper.cpp on :8080.
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()  # loads .env from the repo root if present; env vars still win
except ImportError:
    pass

# Repo root: ~/workspace/interview-sparring-partner/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LLM_PROVIDER = os.getenv("ISP_LLM_PROVIDER", "cloud").strip().lower()  # "cloud" or "ollama"

LLM_CHAIN = [
    p.strip().lower()
    for p in os.getenv("ISP_LLM_CHAIN", "groq,gemini").split(",")
    if p.strip()
]

_CLOUD_PROVIDER_SPECS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "openai/gpt-oss-120b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-3.6-flash",
    },
}

LLM_FAILOVER_COOLDOWN = int(os.getenv("ISP_LLM_COOLDOWN", "60"))

OLLAMA_URL = os.getenv("ISP_OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("ISP_OLLAMA_MODEL", "cyberforge-v2")

WHISPER_URL = os.getenv("WHISPER_URL", "http://127.0.0.1:8080").rstrip("/")
TTS_VOICE = os.getenv("TTS_VOICE", "hi-IN-SwaraNeural")

DATA_DIR = os.getenv("ISP_DATA_DIR", os.path.join(_REPO_ROOT, "data"))
DB_PATH = os.path.join(DATA_DIR, "ispar.db")


def _provider_keys(name: str) -> list[str]:
    """Collect API keys for a provider from {NAME}_API_KEY / {NAME}_API_KEYS (comma-separated)."""
    keys: list[str] = []
    single = os.getenv(f"{name.upper()}_API_KEY", "").strip()
    if single:
        keys.append(single)
    for key in os.getenv(f"{name.upper()}_API_KEYS", "").split(","):
        key = key.strip()
        if key:
            keys.append(key)
    return keys


def cloud_chain() -> list[dict]:
    """Build the ordered list of usable cloud providers.

    Each entry is {"provider", "base_url", "api_key", "model"}. Unknown chain
    names are skipped, and so are providers with no configured API keys.
    """
    chain: list[dict] = []
    for name in LLM_CHAIN:
        spec = _CLOUD_PROVIDER_SPECS.get(name)
        if spec is None:
            continue
        keys = _provider_keys(name)
        if not keys:
            continue
        upper = name.upper()
        base_url = os.getenv(f"{upper}_BASE_URL", spec["base_url"]).rstrip("/")
        model = os.getenv(f"{upper}_MODEL", spec["default_model"]).strip()
        if not model:
            model = spec["default_model"]
        # Every configured key becomes its own chain entry, so one key's
        # rate limit fails over to the next key for the same provider.
        for key in keys:
            chain.append(
                {
                    "provider": name,
                    "base_url": base_url,
                    "api_key": key,
                    "model": model,
                }
            )
    return chain
