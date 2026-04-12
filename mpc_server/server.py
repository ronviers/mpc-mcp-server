"""
mpc_server/server.py

FastMCP-based MPC server with HTTP UI proxy.

MCP transport : stdio  (register in Claude Desktop / any MCP client)
HTTP UI       : http://localhost:7771  (browser reference interface)

MCP tools
---------
  compile_text          Full MPC analysis (hypotheses, phases, compatibility,
                        free energy, spin-glass ground state, Theorem 6.1)
  compile_sequence      Multi-step trace analysis with cross-step entity ledger
  read_claims           Per-claim phase assignment (c/s/k/r)
  budget_estimate       Theorem 6.1 N_max bound — no API call required
  list_available_models Dynamic model catalogue across all five providers

HTTP endpoints
--------------
  GET  /              → index.html
  GET  /models        → JSON model catalogue (env-var keys)
  POST /models        → JSON model catalogue (caller-supplied keys, forced refresh)
  POST /status        → Provider connectivity + key status
  POST /              → MPC actions (compile, compile_sequence, read_claims,
                        budget_estimate, free_energy_surface, ground_state)
  POST /setenv        → Persist an API key to .env
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

import mpc_core
from mpc_core.providers import (
    DEFAULT_MODEL,
    PROVIDER_ENV_VARS,
    ProviderID,
    all_models_catalogue,
    provider_for_model,
    provider_status,
    resolve_api_key,
)
from mpc_core.router import call_model  # noqa: F401 — re-exported for tests
from mpc_core.thermodynamics import free_energy_surface, find_ground_state

log = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent.parent / "static"
_ENV_FILE   = Path(__file__).parent.parent / ".env"
_UI_PORT    = 7771

# ═══════════════════════════════════════════════════════════════════════════════
#  FastMCP application
# ═══════════════════════════════════════════════════════════════════════════════

mcp = FastMCP(
    name="mpc",
    description=(
        "Metastable Propositional Calculus (MPC) engine. "
        "Analyzes the thermodynamic and physical feasibility of logical assertions "
        "across multi-step reasoning. Unlike standard Boolean logic, MPC detects "
        "'epistemic drift' and structural conflicts (k-states) by calculating the "
        "energetic holding costs of maintaining premises over time. "
        "Use this to rigorously verify whether a complex sequence of claims can be "
        "logically maintained together without collapsing into contradiction."
    ),
)

# ── Tool: compile_text ────────────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Full MPC analysis of a text passage. "
        "Extracts propositional hypotheses, assigns MPC phases (c=committed, "
        "s=suspended, k=conflict, r=reset), builds the pairwise frustration matrix, "
        "computes free energy F=−kT ln Z via QuTiP (or classical approximation), "
        "solves the Ising spin-glass ground state, and returns the Theorem 6.1 "
        "sustainable-hypothesis-count bound. "
        "k-states identify where irreconcilable constraints are held simultaneously."
    )
)
def compile_text(
    text: str,
    model: str = DEFAULT_MODEL,
    provider_api_key: str = "",
    E_star: float = 20.0,
    alpha: float = 1.0,
    temperature: float = 1.0,
    solve_ground_state: bool = True,
) -> dict:
    """
    Parameters
    ----------
    text              : Passage to analyse.
    model             : Model name; provider is auto-detected from the name.
    provider_api_key  : API key for the provider.  Omit to use the env var
                        (ANTHROPIC_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY /
                         KIMI_API_KEY).
    E_star            : Metastability budget in k_B T units (default 20.0).
    alpha             : Substrate efficiency ∈ (0,1] (default 1.0).
    temperature       : Effective temperature in k_B T units (default 1.0).
    solve_ground_state: Whether to run the Ising spin-glass solver (default True).
    """
    result = mpc_core.compile(
        text,
        provider_api_key,
        model=model,
        E_star=E_star,
        alpha=alpha,
        temperature=temperature,
        solve_ground_state=solve_ground_state,
    )
    return result.to_dict()

# ── Tool: compile_sequence ────────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Analyse a multi-step reasoning trace using MPC. "
        "Tracks canonical entities across steps via an Entity Ledger, detecting "
        "when the same proposition is maintained, revised, or contradicted over time. "
        "Computes accumulated historical depth (η_i) — the thermodynamic cost of "
        "revising a belief that has been used in downstream inferences. "
        "Ideal for verifying the logical coherence of chain-of-thought reasoning, "
        "argumentative essays, or scientific inference chains."
    )
)
def compile_sequence(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    provider_api_key: str = "",
    E_star: float = 20.0,
    alpha: float = 1.0,
    temperature: float = 1.0,
) -> dict:
    """
    Parameters
    ----------
    texts            : List of reasoning steps (strings), in order.
    model            : Model name; provider auto-detected.
    provider_api_key : API key for the provider.
    E_star           : Metastability budget in k_B T units.
    alpha            : Substrate efficiency ∈ (0,1].
    temperature      : Effective temperature.
    """
    result = mpc_core.compile_sequence(
        texts,
        provider_api_key,
        model=model,
        E_star=E_star,
        alpha=alpha,
        temperature=temperature,
    )
    return result.to_dict()

# ── Tool: read_claims ─────────────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Assign MPC phases to a flat list of claims without building a full "
        "compatibility matrix. Fast and token-efficient. "
        "Returns phase (c/s/k/r), barrier height, linguistic register, and "
        "one-sentence rationale for each claim. "
        "Useful for quickly auditing a list of assumptions or premises."
    )
)
def read_claims(
    claims: list[str],
    model: str = DEFAULT_MODEL,
    provider_api_key: str = "",
    E_c: float = 1.0,
    E_s: float = 3.0,
) -> list[dict]:
    """
    Parameters
    ----------
    claims           : List of propositional strings to classify.
    model            : Model name.
    provider_api_key : API key for the provider.
    E_c              : Committed threshold in k_B T (default 1.0).
    E_s              : Suspended threshold in k_B T (default 3.0).
    """
    return mpc_core.read_claims(
        claims,
        provider_api_key,
        model=model,
        E_c=E_c,
        E_s=E_s,
    )

# ── Tool: budget_estimate ─────────────────────────────────────────────────────

@mcp.tool(
    description=(
        "Compute the Theorem 6.1 sustainable-hypothesis-count bound: "
        "N_max = O(√(2E* / α ε_min d_avg)). "
        "No API call required — pure arithmetic. "
        "Tells you the maximum number of hypotheses a system with the given "
        "parameters can simultaneously maintain without thermodynamic collapse."
    )
)
def budget_estimate(
    N: int,
    d_avg: float,
    epsilon_min: float,
    alpha: float = 1.0,
    E_star: float = 20.0,
) -> dict:
    """
    Parameters
    ----------
    N           : Current number of hypotheses in the system.
    d_avg       : Average constraint degree (mean number of frustrating pairs per node).
    epsilon_min : Minimum non-zero pairwise frustration in k_B T units.
    alpha       : Substrate efficiency ∈ (0,1] (default 1.0).
    E_star      : Metastability budget in k_B T units (default 20.0).
    """
    est = mpc_core.budget_estimate(N, d_avg, epsilon_min, alpha, E_star)
    return {
        "N":              est.N,
        "N_max":          round(est.N_max, 2) if est.N_max != float("inf") else None,
        "margin":         round(est.margin, 2) if est.margin != float("inf") else None,
        "interpretation": est.interpretation,
        "parameters": {
            "d_avg":       est.d_avg,
            "epsilon_min": est.epsilon_min,
            "alpha":       est.alpha,
            "E_star":      est.E_star,
        },
    }

# ── Tool: list_available_models ───────────────────────────────────────────────

@mcp.tool(
    description=(
        "Return the real-time model catalogue across all five providers "
        "(Anthropic, Google, OpenAI, Kimi, Ollama). "
        "Provider APIs are queried dynamically and results are cached for 5 minutes. "
        "Providers for which no API key is configured return their built-in "
        "fallback list. Pass provider_keys to override environment variables."
    )
)
def list_available_models(
    anthropic_key: str = "",
    google_key:    str = "",
    openai_key:    str = "",
    kimi_key:      str = "",
    ollama_host:   str = "",
    force_refresh: bool = False,
) -> dict:
    """
    Returns a dict keyed by provider name, each containing a list of model IDs.
    """
    from mpc_core.providers import list_models, ProviderID
    result = {}
    key_map = {
        ProviderID.ANTHROPIC: anthropic_key,
        ProviderID.GOOGLE:    google_key,
        ProviderID.OPENAI:    openai_key,
        ProviderID.KIMI:      kimi_key,
        ProviderID.OLLAMA:    ollama_host,
    }
    for pid, key in key_map.items():
        result[pid.value] = list_models(pid, key, force_refresh=force_refresh)
    return result

# ═══════════════════════════════════════════════════════════════════════════════
#  HTTP UI server
# ═══════════════════════════════════════════════════════════════════════════════

def _load_env_file():
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

def _save_env_key(key_name: str, key_value: str):
    lines = _ENV_FILE.read_text().splitlines() if _ENV_FILE.exists() else []
    updated = False
    for i, line in enumerate(lines):
        if line.strip().startswith(key_name + "="):
            lines[i] = f"{key_name}={key_value}"
            updated = True
            break
    if not updated:
        lines.append(f"{key_name}={key_value}")
    _ENV_FILE.write_text("\n".join(lines) + "\n")
    os.environ[key_name] = key_value
    log.info("Saved %s to .env", key_name)

def _resolve_api_key_from_payload(model: str, payload: dict) -> str:
    """Pick the right API key from the HTTP request payload."""
    provider = provider_for_model(model, payload.get("ollama_host", ""))
    # Map provider to payload key name
    key_map = {
        ProviderID.ANTHROPIC: "api_key",
        ProviderID.GOOGLE:    "google_api_key",
        ProviderID.OPENAI:    "openai_api_key",
        ProviderID.KIMI:      "kimi_api_key",
        ProviderID.OLLAMA:    "ollama_host",
    }
    payload_key = key_map.get(provider, "api_key")
    explicit = payload.get(payload_key, "")
    return resolve_api_key(provider, explicit)

def _sanitize_json(obj: Any) -> Any:
    """Replace inf/nan with None so json.dumps never raises."""
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


class _UIHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence per-request access log

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    # ── GET ───────────────────────────────────────────────────────────────────

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")

        if path == "/models":
            # Return model catalogue using env-var keys (no body needed)
            data = json.dumps(_sanitize_json(all_models_catalogue())).encode()
            self._respond(200, "application/json", data)
            return

        if path == "/env":
            # Return which env vars are currently set (keys redacted)
            status = {
                env: "set" if os.environ.get(env) else "unset"
                for env in PROVIDER_ENV_VARS.values()
            }
            self._respond(200, "application/json", json.dumps(status).encode())
            return

        # Static files
        file_name = self.path.lstrip("/") or "index.html"
        file_path = _STATIC_DIR / file_name
        if file_path.exists() and file_path.is_file():
            data = file_path.read_bytes()
            ct = "text/html" if file_name.endswith(".html") else "application/octet-stream"
            self._respond(200, ct, data)
        else:
            self.send_error(404)

    # ── POST ──────────────────────────────────────────────────────────────────

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        path   = self.path.rstrip("/")

        # /setenv — persist a single key=value to .env
        if path == "/setenv":
            try:
                payload   = json.loads(body)
                key_name  = payload.get("key_name",  "")
                key_value = payload.get("key_value", "").strip()
                if not key_name:
                    self._json_error(400, "key_name is required")
                    return
                if key_value:
                    _save_env_key(key_name, key_value)
                    out = {"ok": True, "key_name": key_name}
                else:
                    out = {"ok": False, "error": "empty value — key not saved"}
                self._respond(200, "application/json", json.dumps(out).encode())
            except Exception as exc:
                self._json_error(500, str(exc))
            return

        # /models (POST) — refresh catalogue with caller-supplied keys
        if path == "/models":
            try:
                payload = json.loads(body) if body else {}
                catalogue = all_models_catalogue(
                    anthropic_key = payload.get("anthropic_key", ""),
                    google_key    = payload.get("google_key",    ""),
                    openai_key    = payload.get("openai_key",    ""),
                    kimi_key      = payload.get("kimi_key",      ""),
                    ollama_host   = payload.get("ollama_host",   ""),
                )
                self._respond(
                    200, "application/json",
                    json.dumps(_sanitize_json(catalogue)).encode(),
                )
            except Exception as exc:
                self._json_error(500, str(exc))
            return

        # /status — provider health check
        if path == "/status":
            try:
                self._respond(
                    200, "application/json",
                    json.dumps(_sanitize_json(provider_status())).encode(),
                )
            except Exception as exc:
                self._json_error(500, str(exc))
            return

        # / — MPC actions
        try:
            payload = json.loads(body)
            action  = payload.get("action")
            model   = payload.get("model", DEFAULT_MODEL)
            api_key = _resolve_api_key_from_payload(model, payload)

            kwargs = dict(
                model=model,
                api_key=api_key,
                E_star=float(payload.get("E_star", 20.0)),
                alpha=float(payload.get("alpha",  1.0)),
                temperature=float(payload.get("temperature", 1.0)),
            )

            response_data = self._handle_action(action, payload, kwargs)
            if response_data is None:
                self._json_error(400, f"Unknown action: {action!r}")
                return
            self._respond(
                200, "application/json",
                json.dumps(_sanitize_json(response_data)).encode(),
            )

        except Exception as exc:
            log.exception("UI proxy error (action=%s)", payload.get("action") if body else "?")
            self._json_error(500, str(exc))

    def _handle_action(self, action: str, payload: dict, kwargs: dict):
        model   = kwargs["model"]
        api_key = kwargs["api_key"]
        E_star  = kwargs["E_star"]
        alpha   = kwargs["alpha"]
        temp    = kwargs["temperature"]

        if action == "compile":
            result = mpc_core.compile(
                payload["text"], api_key,
                model=model, E_star=E_star, alpha=alpha, temperature=temp,
            )
            return result.to_dict()

        if action == "compile_sequence":
            result = mpc_core.compile_sequence(
                payload["texts"], api_key,
                model=model, E_star=E_star, alpha=alpha, temperature=temp,
            )
            return result.to_dict()

        if action == "read_claims":
            return mpc_core.read_claims(
                payload["claims"], api_key,
                model=model,
                E_c=float(payload.get("E_c", 1.0)),
                E_s=float(payload.get("E_s", 3.0)),
            )

        if action == "budget_estimate":
            est = mpc_core.budget_estimate(
                payload["N"], payload["d_avg"], payload["epsilon_min"],
                float(payload.get("alpha", 1.0)), float(payload.get("E_star", 20.0)),
            )
            return {
                "N": est.N,
                "N_max":          est.N_max  if est.N_max  != float("inf") else None,
                "margin":         est.margin if est.margin != float("inf") else None,
                "interpretation": est.interpretation,
            }

        if action == "free_energy_surface":
            if "epsilon_matrix" in payload:
                raw_eps = payload["epsilon_matrix"]
            elif "text" in payload:
                r = mpc_core.compile(payload["text"], api_key, model=model)
                hyps = r.hypotheses
                ids  = [h.id for h in hyps]
                idx  = {hid: i for i, hid in enumerate(ids)}
                N    = len(hyps)
                raw_eps = [[0.0]*N for _ in range(N)]
                for e in r.compatibility_matrix:
                    if e.hi in idx and e.hj in idx:
                        i, j = idx[e.hi], idx[e.hj]
                        raw_eps[i][j] = raw_eps[j][i] = e.epsilon
            else:
                return {"error": "Provide 'text' or 'epsilon_matrix'"}

            surface = free_energy_surface(
                raw_eps,
                n_T=int(payload.get("n_T", 20)),
                n_E=int(payload.get("n_E", 20)),
            )
            return surface

        if action == "ground_state":
            gs = find_ground_state(payload.get("epsilon_matrix", []))
            return gs

        return None  # unknown action

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _respond(self, code: int, content_type: str, data: bytes):
        self.send_response(code)
        self.send_header("Content-Type",   content_type)
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _json_error(self, code: int, msg: str):
        data = json.dumps({"error": msg}).encode()
        self._respond(code, "application/json", data)


def _start_http_server():
    httpd = HTTPServer(("127.0.0.1", _UI_PORT), _UIHandler)
    log.info("MPC reference UI → http://localhost:%d", _UI_PORT)
    print(f"MPC reference UI → http://localhost:{_UI_PORT}", flush=True)
    httpd.serve_forever()

# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _load_env_file()

    ui_thread = threading.Thread(target=_start_http_server, daemon=True)
    ui_thread.start()

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
