"""
mpc_core/json_repair.py

Best-effort repair of truncated JSON produced by LLMs that hit a token limit.

Strategy
--------
1. Try stdlib json.loads() — return immediately on success.
2. Strip markdown code fences and retry.
3. Call _close_json() to structurally close open strings/arrays/objects, then
   iteratively back up to the last clean element boundary if parsing still fails.
4. Fall back to _extract_partial() — regex-extract the hypotheses array.
5. Raise ValueError only if nothing parseable remains.
"""
from __future__ import annotations

import json
import re
from typing import Any

_REQUIRED_KEYS: dict[str, Any] = {
    "hypotheses":           [],
    "compatibility_matrix": [],
    "energy_model": {
        "total_load": 0.0,
        "budget_utilization": 0.0,
        "dominant_phase": "s",
    },
    "k_states":            [],
    "analytical_summary":  "(response truncated — JSON was incomplete)",
}

_MAX_RETRIES = 12


def loads_or_repair(text: str) -> dict:
    """
    Parse *text* as JSON.  On failure, attempt structural repair.
    Always returns a dict with all required MPC keys present.
    Raises ValueError only if no dict is recoverable at all.
    """
    # 1. Fast path
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return _fill_required_keys(result)
        raise ValueError(f"Expected JSON object, got {type(result).__name__}")
    except json.JSONDecodeError:
        pass

    # 2. Strip fences
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return _fill_required_keys(result)
    except json.JSONDecodeError:
        pass

    # 3. Structural repair with iterative back-up
    result = _repair_with_backup(text)
    if result is not None:
        result = _fill_required_keys(result)
        result["_json_repaired"] = True
        return result

    # 4. Partial extraction
    partial = _extract_partial(text)
    if partial is not None:
        partial = _fill_required_keys(partial)
        partial["_json_repaired"] = True
        partial["_json_partial"]  = True
        return partial

    raise ValueError(
        "JSON repair failed — response may be far too short or malformed.\n"
        f"First 200 chars: {text[:200]!r}"
    )


def _repair_with_backup(text: str) -> dict | None:
    candidate = text
    for _ in range(_MAX_RETRIES):
        closed = _close_json(candidate)
        try:
            result = json.loads(closed)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        shorter = _trim_to_last_clean_boundary(candidate)
        if shorter is None or len(shorter) >= len(candidate):
            break
        candidate = shorter

    return None


def _close_json(text: str) -> str:
    stack:       list[str] = []
    in_string:   bool      = False
    escape_next: bool      = False
    result:      list[str] = []

    for ch in text:
        if escape_next:
            escape_next = False
            result.append(ch)
            continue

        if ch == "\\" and in_string:
            escape_next = True
            result.append(ch)
            continue

        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue

        if in_string:
            if ch in ("\n", "\r", "\t"):
                result.append("\\n" if ch == "\n" else "\\r" if ch == "\r" else "\\t")
            else:
                result.append(ch)
            continue

        if ch in ("{", "["):
            stack.append(ch)
        elif ch in ("}", "]"):
            if stack:
                stack.pop()

        result.append(ch)

    out = "".join(result)

    if in_string:
        out += '"'

    out = _strip_incomplete_last_token(out)
    out = re.sub(r",(\s*)$", r"\1", out.rstrip())

    for opener in reversed(stack):
        out += "}" if opener == "{" else "]"

    return out


def _strip_incomplete_last_token(s: str) -> str:
    m = re.search(r',\s*"[^"]*"?\s*(?::\s*[^,}\]]*)?$', s)
    if m:
        return s[:m.start()]
    return s


def _trim_to_last_clean_boundary(text: str) -> str | None:
    in_string   = False
    escape_next = False
    last_boundary = -1

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in (",", "[", "{"):
            last_boundary = i

    return text[:last_boundary] if last_boundary >= 0 else None


def _extract_partial(text: str) -> dict | None:
    m = re.search(r'"hypotheses"\s*:\s*\[', text)
    if not m:
        return None
    try:
        wrapped  = "{" + text[m.start():]
        repaired = _close_json(wrapped)
        result   = json.loads(repaired)
        if isinstance(result, dict) and "hypotheses" in result:
            return result
    except Exception:
        pass
    return None


def _fill_required_keys(d: dict) -> dict:
    for key, default in _REQUIRED_KEYS.items():
        if key not in d:
            d[key] = default
    return d
