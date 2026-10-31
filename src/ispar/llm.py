"""Talking to the LLM — local Ollama, or the cloud chain from config.

Why a chain at all: in Oct 2026 three Groq keys in a row died on me with
401 "invalid API key", so Gemini became primary and Groq the fallback.
One rule from day one, and it stays: failures raise LLMError, never a
canned reply. A fake evaluation would ruin the whole point of practice.

Update, end of Oct 2026: a working Groq key turned up after all, so Groq
(openai/gpt-oss-120b) is primary again and Gemini the fallback. The order
lives in config / ISP_LLM_CHAIN, not in this file.
"""

from __future__ import annotations

import time

import requests

from . import config

_TIMEOUT = 120

# provider name -> unix timestamp until which it is in failover cooldown
_cooldowns: dict[str, float] = {}


class LLMError(Exception):
    """LLM call failed. The message always names the provider and the reason."""


class _Failover(Exception):
    """Retryable failure (429/5xx/network) — the chain moves on to the next provider."""


def chat(messages: list[dict], temperature: float = 0.7, json_mode: bool = False) -> str:
    """One chat call. Returns the assistant's text; raises LLMError on failure."""
    if config.LLM_PROVIDER == "ollama":
        return _chat_ollama(messages, temperature, json_mode)
    return _chat_cloud(messages, temperature, json_mode)


def _chat_ollama(messages: list[dict], temperature: float, json_mode: bool) -> str:
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"
    url = f"{config.OLLAMA_URL}/api/chat"
    try:
        resp = requests.post(url, json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise LLMError(f"ollama request failed: {exc}") from exc
    except ValueError as exc:
        raise LLMError(f"ollama returned invalid JSON: {exc}") from exc
    try:
        content = data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise LLMError(f"ollama returned unexpected response shape: {exc}") from exc
    return str(content).strip()


def _chat_cloud(messages: list[dict], temperature: float, json_mode: bool) -> str:
    chain = config.cloud_chain()
    if not chain:
        raise LLMError(
            "cloud LLM failed: no providers configured "
            "(ISP_LLM_CHAIN has no entries with API keys; set e.g. GROQ_API_KEY)"
        )
    failures: list[str] = []
    for entry in chain:
        provider = entry["provider"]
        if _cooldowns.get(provider, 0.0) > time.time():
            continue
        try:
            return _cloud_request(entry, messages, temperature, json_mode)
        except LLMError:
            raise  # fatal (400/401 etc.): abort the whole chain immediately
        except _Failover as exc:
            # 429 / 5xx / network error: cooldown recorded, try the next entry
            _cooldowns[provider] = time.time() + config.LLM_FAILOVER_COOLDOWN
            failures.append(f"{provider} failed: {exc}")
            continue
    if failures:
        raise LLMError("cloud LLM failed: " + "; ".join(failures))
    raise LLMError("cloud LLM failed: all configured providers are in cooldown")


def _cloud_request(
    entry: dict, messages: list[dict], temperature: float, json_mode: bool
) -> str:
    provider = entry["provider"]
    payload = {
        "model": entry["model"],
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    url = f"{entry['base_url']}/chat/completions"
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {entry['api_key']}"},
            timeout=_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise _Failover(f"network error: {exc}") from exc
    if resp.status_code == 429 or 500 <= resp.status_code < 600:
        raise _Failover(
            f"HTTP {resp.status_code} ({resp.text[:200].strip()})"
        )
    if resp.status_code in (400, 401):
        # Groq's gpt-oss models sometimes answer a json_mode request with
        # 400 "json_validate_failed" — the model failed to produce JSON,
        # which is a per-model capability failure, not a bad request.
        # Fall over to the next provider instead of killing the whole
        # chain (found live on Render, 31 Oct 2026, Groq as primary).
        # Every other 400, and all 401s, stay fatal.
        if resp.status_code == 400 and "json_validate_failed" in resp.text:
            raise _Failover(
                f"HTTP 400 json_validate_failed ({resp.text[:200].strip()})"
            )
        raise LLMError(
            f"{provider} request failed: HTTP {resp.status_code} "
            f"({resp.text[:200].strip()})"
        )
    if not resp.ok:
        raise LLMError(
            f"{provider} request failed: HTTP {resp.status_code} "
            f"({resp.text[:200].strip()})"
        )
    try:
        data = resp.json()
    except ValueError as exc:
        raise LLMError(f"{provider} returned invalid JSON: {exc}") from exc
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"{provider} returned unexpected response shape: {exc}") from exc
    return str(content).strip()
