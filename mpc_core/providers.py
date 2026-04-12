"""
mpc_core/providers.py

Provider registry for MPC model routing.

Supported providers
-------------------
  anthropic  — Anthropic Claude  (anthropic SDK, /v1/models)
  google     — Google Gemini     (google-generativeai SDK, list_models())
  openai     — OpenAI GPT / o-series  (openai SDK, /v1/models)
  kimi       — Moonshot AI Kimi  (openai SDK, moonshot.cn base_url)
  ollama     — Local Ollama      (REST /api/tags — no key needed)

API key resolution order (per provider)
----------------------------------------
  1. Explicit api_key argument passed at call time
  2. Environment variable  (see PROVIDER_ENV_VARS below)
  3. Empty string → SDK will raise a clear auth error

Dynamic model listing is cached per (provider, key-prefix) with a 5-minute TTL.
On any list error the last known list (or built-in fallback) is returned so the
UI never shows an empty dropdown.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)

# ── Provider identifiers ──────────────────────────────────────────────────────

class ProviderID(str, Enum):
    ANTHROPIC = "anthropic"
    GOOGLE    = "google"
    OPENAI    = "openai"
    KIMI      = "kimi"
    OLLAMA    = "ollama"

# ── Per-provider environment variable names ───────────────────────────────────
# Each provider has exactly one env var; no ambiguity, no conflicts.

PROVIDER_ENV_VARS: dict[ProviderID, str] = {
    ProviderID.ANTHROPIC: "ANTHROPIC_API_KEY",
    ProviderID.GOOGLE:    "GOOGLE_API_KEY",
    ProviderID.OPENAI:    "OPENAI_API_KEY",
    ProviderID.KIMI:      "KIMI_API_KEY",
    ProviderID.OLLAMA:    "OLLAMA_HOST",       # host URL, not a secret
}

# ── Provider-specific constants ───────────────────────────────────────────────

KIMI_BASE_URL = "https://api.moonshot.cn/v1"

OLLAMA_DEFAULT_HOST = "http://localhost:11434"

# Built-in fallback model lists (used when provider API is unreachable or key absent)
_FALLBACKS: dict[ProviderID, list[str]] = {
    ProviderID.ANTHROPIC: [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ],
    ProviderID.GOOGLE: [
        "gemini-2.5-pro-preview-03-25",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
    ProviderID.OPENAI: [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "o1",
        "o3-mini",
    ],
    ProviderID.KIMI: [
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshot-v1-128k",
    ],
    ProviderID.OLLAMA: [],
}

# Default model per provider
PROVIDER_DEFAULTS: dict[ProviderID, str] = {
    ProviderID.ANTHROPIC: "claude-sonnet-4-6",
    ProviderID.GOOGLE:    "gemini-2.0-flash",
    ProviderID.OPENAI:    "gpt-4o",
    ProviderID.KIMI:      "moonshot-v1-32k",
    ProviderID.OLLAMA:    "",
}

DEFAULT_MODEL    = PROVIDER_DEFAULTS[ProviderID.ANTHROPIC]
DEFAULT_PROVIDER = ProviderID.ANTHROPIC

# ── Model cache ───────────────────────────────────────────────────────────────
_CACHE_TTL_SEC = 300  # 5 minutes

@dataclass
class _CacheEntry:
    models: list[str]
    ts: float = field(default_factory=time.monotonic)

    def fresh(self) -> bool:
        return time.monotonic() - self.ts < _CACHE_TTL_SEC

_model_cache: dict[str, _CacheEntry] = {}

# ── Provider detection ────────────────────────────────────────────────────────

def provider_for_model(model: str, ollama_host: str = "") -> ProviderID:
    """
    Infer the provider from the model name string.
    Ollama is checked last via a live /api/tags query (fast, cached).
    """
    m = model.lower().strip()
    if m.startswith(("claude-", "claude_")):
        return ProviderID.ANTHROPIC
    if m.startswith(("gemini-", "gemini_")):
        return ProviderID.GOOGLE
    if m.startswith(("gpt-", "o1", "o1-", "o3", "o3-", "o4", "o4-")):
        return ProviderID.OPENAI
    if m.startswith(("moonshot-", "kimi-")):
        return ProviderID.KIMI

    # Check Ollama last — may involve a network call (but cached)
    host = ollama_host or os.environ.get("OLLAMA_HOST", OLLAMA_DEFAULT_HOST)
    installed = _list_ollama_raw(host)
    installed_lower = {i.lower() for i in installed}
    base = m.split(":")[0]
    if m in installed_lower or base in installed_lower or any(
        i.startswith(base) for i in installed_lower
    ):
        return ProviderID.OLLAMA

    log.debug("Cannot detect provider for model %r — defaulting to anthropic", model)
    return ProviderID.ANTHROPIC

# ── API key resolution ────────────────────────────────────────────────────────

def resolve_api_key(provider: ProviderID, explicit_key: str = "") -> str:
    """
    Return the API key (or host URL) for a provider.

    Priority: explicit_key → environment variable → empty string.

    For OLLAMA the return value is the host URL, not a secret key.
    For all other providers an empty return means the SDK will raise
    an AuthenticationError — surfaced as a clear error to the user.
    """
    if explicit_key:
        return explicit_key
    env_var = PROVIDER_ENV_VARS.get(provider, "")
    if provider == ProviderID.OLLAMA:
        return os.environ.get("OLLAMA_HOST", OLLAMA_DEFAULT_HOST)
    return os.environ.get(env_var, "")

# ── Dynamic model listing ─────────────────────────────────────────────────────

def list_models(
    provider: ProviderID,
    api_key: str = "",
    *,
    force_refresh: bool = False,
) -> list[str]:
    """
    Return available models for *provider*, using *api_key* (or env var fallback).

    Results are cached.  On any error the built-in fallback list is returned
    so callers never receive an empty list unexpectedly.
    """
    # Cache key uses provider + first 8 chars of key so different keys get
    # different cache entries without storing the full secret.
    cache_key = f"{provider.value}:{(api_key or '')[:8]}"
    if (
        not force_refresh
        and cache_key in _model_cache
        and _model_cache[cache_key].fresh()
    ):
        return _model_cache[cache_key].models

    fetched = _fetch_models_safe(provider, api_key)
    _model_cache[cache_key] = _CacheEntry(models=fetched)
    return fetched

def _fetch_models_safe(provider: ProviderID, api_key: str) -> list[str]:
    """Fetch models; always return a non-empty list (uses fallback on error)."""
    try:
        result = _fetch_models(provider, api_key)
        if result:
            return result
    except Exception as exc:
        log.warning("Model listing failed for %s: %s", provider.value, exc)
    return _FALLBACKS.get(provider, [])

def _fetch_models(provider: ProviderID, api_key: str) -> list[str]:
    if provider == ProviderID.ANTHROPIC:
        return _list_anthropic(api_key)
    if provider == ProviderID.GOOGLE:
        return _list_google(api_key)
    if provider == ProviderID.OPENAI:
        return _list_openai(api_key)
    if provider == ProviderID.KIMI:
        return _list_kimi(api_key)
    if provider == ProviderID.OLLAMA:
        host = api_key or os.environ.get("OLLAMA_HOST", OLLAMA_DEFAULT_HOST)
        return _list_ollama_raw(host)
    return []

# ── Per-provider listing implementations ─────────────────────────────────────

def _list_anthropic(api_key: str) -> list[str]:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        log.debug("No ANTHROPIC_API_KEY — returning fallback list")
        return _FALLBACKS[ProviderID.ANTHROPIC]
    import anthropic  # lazy import
    client = anthropic.Anthropic(api_key=key)
    page = client.models.list(limit=100)
    # models.list() returns a SyncPage; iterate to collect all
    ids = [m.id for m in page]
    # Paginate if there are more
    while page.has_next_page():
        page = page.get_next_page()
        ids.extend(m.id for m in page)
    return sorted(ids, reverse=True) if ids else _FALLBACKS[ProviderID.ANTHROPIC]

def _list_google(api_key: str) -> list[str]:
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        return _FALLBACKS[ProviderID.GOOGLE]
    import google.generativeai as genai  # lazy import
    genai.configure(api_key=key)
    models = [
        m.name.removeprefix("models/")
        for m in genai.list_models()
        if "generateContent" in getattr(m, "supported_generation_methods", [])
    ]
    gemini = sorted([m for m in models if m.startswith("gemini")], reverse=True)
    return gemini or _FALLBACKS[ProviderID.GOOGLE]

def _list_openai(api_key: str) -> list[str]:
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return _FALLBACKS[ProviderID.OPENAI]
    import openai  # lazy import
    client = openai.OpenAI(api_key=key)
    all_ids = [m.id for m in client.models.list().data]
    # Keep only chat-capable model families
    prefixes = ("gpt-4", "gpt-3.5", "o1", "o3", "o4", "chatgpt")
    chat = sorted(
        [m for m in all_ids if any(m.startswith(p) for p in prefixes)],
        reverse=True,
    )
    return chat or _FALLBACKS[ProviderID.OPENAI]

def _list_kimi(api_key: str) -> list[str]:
    key = api_key or os.environ.get("KIMI_API_KEY", "")
    if not key:
        return _FALLBACKS[ProviderID.KIMI]
    import openai  # lazy import — Kimi is OpenAI-compatible
    client = openai.OpenAI(api_key=key, base_url=KIMI_BASE_URL)
    all_ids = [m.id for m in client.models.list().data]
    moonshot = sorted([m for m in all_ids if "moonshot" in m.lower()])
    return moonshot or _FALLBACKS[ProviderID.KIMI]

def _list_ollama_raw(host: str) -> list[str]:
    """Query a running Ollama instance for its installed models."""
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []

# ── Full catalogue ────────────────────────────────────────────────────────────

def all_models_catalogue(
    *,
    anthropic_key: str = "",
    google_key:    str = "",
    openai_key:    str = "",
    kimi_key:      str = "",
    ollama_host:   str = "",
) -> dict[str, list[str]]:
    """
    Return the full model catalogue across all five providers.

    Keys match ProviderID values: "anthropic", "google", "openai", "kimi", "ollama".
    Each provider is queried independently; failures degrade gracefully to
    the built-in fallback list for that provider.
    """
    host = ollama_host or os.environ.get("OLLAMA_HOST", OLLAMA_DEFAULT_HOST)
    return {
        ProviderID.ANTHROPIC.value: list_models(ProviderID.ANTHROPIC, anthropic_key),
        ProviderID.GOOGLE.value:    list_models(ProviderID.GOOGLE,    google_key),
        ProviderID.OPENAI.value:    list_models(ProviderID.OPENAI,    openai_key),
        ProviderID.KIMI.value:      list_models(ProviderID.KIMI,      kimi_key),
        ProviderID.OLLAMA.value:    list_models(ProviderID.OLLAMA,    host),
    }

def provider_status() -> dict[str, dict]:
    """
    Return connectivity status and model count for each provider.
    Used by the /status endpoint for the UI health panel.
    """
    status: dict[str, dict] = {}
    for pid in ProviderID:
        key = resolve_api_key(pid)
        try:
            models = list_models(pid, key)
            status[pid.value] = {
                "available": True,
                "model_count": len(models),
                "key_set": bool(key and pid != ProviderID.OLLAMA),
                "env_var": PROVIDER_ENV_VARS.get(pid, ""),
            }
        except Exception as exc:
            status[pid.value] = {
                "available": False,
                "model_count": 0,
                "key_set": bool(key),
                "env_var": PROVIDER_ENV_VARS.get(pid, ""),
                "error": str(exc),
            }
    return status
