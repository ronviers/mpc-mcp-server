"""
mpc_core/router.py

Unified model router: routes (system, user) prompts to the correct provider SDK
and returns plain text.  Provider detection, key resolution, and model listing
are delegated to mpc_core.providers.

Retry logic
-----------
Transient errors (rate limits, timeouts) are retried up to MAX_RETRIES times
with exponential back-off.  Authentication errors fail immediately.

Streaming
---------
All calls use non-streaming for simplicity; streaming can be added per-provider.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Optional

from .providers import (
    KIMI_BASE_URL,
    OLLAMA_DEFAULT_HOST,
    DEFAULT_MODEL,
    ProviderID,
    all_models_catalogue,
    list_models,
    provider_for_model,
    resolve_api_key,
)

log = logging.getLogger(__name__)

MAX_RETRIES    = 3
RETRY_BASE_SEC = 1.5   # first retry after ~1.5 s, doubles each time

# ── Public surface ────────────────────────────────────────────────────────────

def call_model(
    system:       str,
    user:         str,
    model:        str = DEFAULT_MODEL,
    api_key:      str = "",
    max_tokens:   int = 4096,
    ollama_host:  str = "",
) -> str:
    """
    Route a (system, user) prompt to the correct backend and return text.

    api_key is the key for whatever provider the model belongs to.
    If omitted, the appropriate environment variable is used automatically.

    Raises RuntimeError with a user-legible message on final failure.
    """
    provider    = provider_for_model(model, ollama_host)
    resolved_key = resolve_api_key(provider, api_key)

    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _dispatch(system, user, model, provider, resolved_key,
                             max_tokens, ollama_host)
        except _AuthError:
            raise   # authentication errors are not retried
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                sleep = RETRY_BASE_SEC * (2 ** (attempt - 1))
                log.warning(
                    "call_model attempt %d/%d failed (%s) — retrying in %.1fs",
                    attempt, MAX_RETRIES, exc, sleep,
                )
                time.sleep(sleep)

    raise RuntimeError(
        f"call_model failed after {MAX_RETRIES} attempts for model {model!r}: {last_exc}"
    ) from last_exc

# ── Dispatch ──────────────────────────────────────────────────────────────────

def _dispatch(
    system: str,
    user: str,
    model: str,
    provider: ProviderID,
    api_key: str,
    max_tokens: int,
    ollama_host: str,
) -> str:
    if provider == ProviderID.ANTHROPIC:
        return _call_anthropic(system, user, model, api_key, max_tokens)
    if provider == ProviderID.GOOGLE:
        return _call_google(system, user, model, api_key, max_tokens)
    if provider == ProviderID.OPENAI:
        return _call_openai(system, user, model, api_key, max_tokens)
    if provider == ProviderID.KIMI:
        return _call_kimi(system, user, model, api_key, max_tokens)
    if provider == ProviderID.OLLAMA:
        host = (
            ollama_host
            or (api_key if api_key.startswith("http") else "")
            or OLLAMA_DEFAULT_HOST
        )
        return _call_ollama(system, user, model, host, max_tokens)
    raise ValueError(f"Unknown provider: {provider}")

# ── Provider backends ─────────────────────────────────────────────────────────

def _call_anthropic(
    system: str,
    user: str,
    model: str,
    api_key: str,
    max_tokens: int,
) -> str:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic SDK not installed — run: pip install anthropic")
    if not api_key:
        raise _AuthError("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text

def _call_google(
    system: str,
    user: str,
    model: str,
    api_key: str,
    max_tokens: int,
) -> str:
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError(
            "google-generativeai SDK not installed — run: pip install google-generativeai"
        )
    if not api_key:
        raise _AuthError("GOOGLE_API_KEY is not set")
    genai.configure(api_key=api_key)
    g_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system,
        generation_config={"max_output_tokens": max_tokens},
    )
    response = g_model.generate_content(user)
    return response.text

def _call_openai(
    system: str,
    user: str,
    model: str,
    api_key: str,
    max_tokens: int,
) -> str:
    try:
        import openai
    except ImportError:
        raise RuntimeError("openai SDK not installed — run: pip install openai")
    if not api_key:
        raise _AuthError("OPENAI_API_KEY is not set")
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content

def _call_kimi(
    system: str,
    user: str,
    model: str,
    api_key: str,
    max_tokens: int,
) -> str:
    """Kimi (Moonshot AI) uses an OpenAI-compatible API at a custom base_url."""
    try:
        import openai
    except ImportError:
        raise RuntimeError("openai SDK not installed — run: pip install openai")
    if not api_key:
        raise _AuthError("KIMI_API_KEY is not set")
    client = openai.OpenAI(api_key=api_key, base_url=KIMI_BASE_URL)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content

def _call_ollama(
    system: str,
    user: str,
    model: str,
    host: str,
    max_tokens: int,
) -> str:
    """Call a locally running Ollama instance."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
        return data["message"]["content"]
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {host}. Is Ollama running? ({exc})"
        ) from exc

# ── Sentinel for non-retryable errors ─────────────────────────────────────────

class _AuthError(RuntimeError):
    """Raised when an API key is missing or rejected. Never retried."""
