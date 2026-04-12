# MPC — Metastable Propositional Calculus

> **Metastable Propositional Calculus (MPC) engine.** Analyzes the thermodynamic and physical feasibility of logical assertions across multi-step reasoning. Unlike standard Boolean logic, MPC detects *epistemic drift* and structural conflicts (k-states) by calculating the energetic holding costs of maintaining premises over time. Use this to rigorously verify whether a complex sequence of claims can be logically maintained together without collapsing into contradiction.

Truth values: **c** (committed) · **s** (suspended) · **k** (conflict) · **r** (reset)

---

## What's new in v0.3

| Feature | Detail |
|---|---|
| **FastMCP** | Replaced low-level MCP server with FastMCP for cleaner tool definitions and better client compatibility |
| **Five-provider routing** | Anthropic · Google · OpenAI · Kimi (Moonshot AI) · Ollama — all first-class |
| **Dynamic model listing** | Real-time queries to each provider's list-models endpoint; 5-minute TTL cache |
| **Zero env-var conflicts** | Each provider has its own variable: `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `KIMI_API_KEY`, `OLLAMA_HOST` |
| **Provider tab in UI** | Side-by-side key configuration, per-provider status, model counts |
| **`list_available_models` tool** | New MCP tool for dynamic model discovery |
| **Retry logic** | Transient errors retried up to 3× with exponential back-off |
| **`/status` + `/env` endpoints** | Provider health and env-var audit without exposing key values |

---

## Providers

| Provider | SDK | Key Env Var | Notes |
|---|---|---|---|
| **Anthropic** | `anthropic` | `ANTHROPIC_API_KEY` | Claude claude-opus-4-6, claude-sonnet-4-6, Haiku |
| **Google** | `google-generativeai` | `GOOGLE_API_KEY` | Gemini 2.5 Pro, 2.0 Flash, 1.5 |
| **OpenAI** | `openai` | `OPENAI_API_KEY` | GPT-4o, o1, o3-mini, etc. |
| **Kimi** | `openai` (OpenAI-compatible) | `KIMI_API_KEY` | Moonshot AI · moonshot.cn |
| **Ollama** | stdlib `urllib` | `OLLAMA_HOST` | Local models, no key needed |

API keys are resolved in order: **explicit argument → environment variable → .env file**. No provider ever reads another provider's key variable.

---

## Install

```bash
# Core — all five providers included
pip install -e .

# Add QuTiP for exact partition function (Ising Hamiltonian)
pip install qutip

# Add NetKet for spin-glass ground-state solver (requires JAX)
pip install "jax[cpu]" netket

# Full stack
pip install -e ".[full]"
```

Requires **Python ≥ 3.11**.

---

## Quick start

### Set API keys

```bash
# Any combination — only set what you have
export ANTHROPIC_API_KEY=sk-ant-…
export GOOGLE_API_KEY=AIza…
export OPENAI_API_KEY=sk-…
export KIMI_API_KEY=sk-…          # Moonshot AI key
export OLLAMA_HOST=http://localhost:11434   # default; omit if standard
```

Or use the **Providers** tab in the browser UI to enter and persist keys to `.env`.

### Start the server

```bash
mpc-server
```

Starts:
- **MCP server** on stdio (register in Claude Desktop or any MCP client)
- **Browser UI** on `http://localhost:7771`

---

## Claude Desktop configuration

```json
{
  "mcpServers": {
    "mpc": {
      "command": "mpc-server",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-…",
        "GOOGLE_API_KEY": "AIza…",
        "OPENAI_API_KEY": "sk-…",
        "KIMI_API_KEY": "sk-…"
      }
    }
  }
}
```

---

## MCP Tools

| Tool | Description | API call? |
|---|---|---|
| `compile_text` | Full MPC analysis: hypotheses, phases, frustration matrix, free energy (QuTiP), spin-glass ground state, Theorem 6.1 bound | ✓ |
| `compile_sequence` | Multi-step trace with Entity Ledger — tracks epistemic drift and η_i accumulation | ✓ |
| `read_claims` | Per-claim phase assignment (c/s/k/r) with rationale | ✓ |
| `budget_estimate` | Theorem 6.1 N_max = O(√(2E*/αε_min d_avg)) — pure arithmetic | ✗ |
| `list_available_models` | Real-time model catalogue across all five providers | ✗ (cached) |

All tools accept `provider_api_key` as an optional parameter; omitting it reads from environment variables automatically.

---

## Python API

```python
import mpc_core

# Full analysis — Anthropic (default)
result = mpc_core.compile("Your text here…", api_key="sk-ant-…")

# Google Gemini
result = mpc_core.compile("…", model="gemini-2.0-flash", api_key="AIza…")

# OpenAI
result = mpc_core.compile("…", model="gpt-4o", api_key="sk-…")

# Kimi (Moonshot AI)
result = mpc_core.compile("…", model="moonshot-v1-32k", api_key="sk-…")

# Local Ollama (no key needed)
result = mpc_core.compile("…", model="llama3:8b")

print(result.energy_model.free_energy)      # F = -kT ln Z
print(result.ground_state.energy)           # Ising ground-state energy
print(result.ground_state.stable_ids)       # most compatible hypothesis subset
print(result.analytical_summary)

# Per-claim phase assignment
phases = mpc_core.read_claims(
    ["All ravens are black.", "Some ravens are albino."],
    api_key="sk-ant-…",
    model="claude-sonnet-4-6",
)

# Budget theorem (no API call)
est = mpc_core.budget_estimate(N=8, d_avg=2.5, epsilon_min=1.2)
print(est.interpretation)

# Multi-step trace with Entity Ledger
seq = mpc_core.compile_sequence(
    ["Step 1 text…", "Step 2 text…", "Step 3 text…"],
    api_key="sk-ant-…",
)

# Dynamic model listing
from mpc_core.providers import list_models, ProviderID
models = list_models(ProviderID.ANTHROPIC, "sk-ant-…")

# Free-energy surface (no API call)
from mpc_core.thermodynamics import free_energy_surface
surface = free_energy_surface(my_epsilon_matrix, T_range=(0.2,5), E_star_range=(2,40))
# surface["F"][T_index][E_star_index] → F value
```

---

## Browser UI — tabs

| Tab | Description |
|---|---|
| **Analyse text** | Full MPC analysis with hypothesis cards, thermodynamic strip (Z, F, S), ground-state box |
| **Read claims** | One claim per line → instant phase assignment |
| **3D Free Energy** | Interactive Plotly F(T, E*) surface with N_max Theorem 6.1 contour |
| **Energy landscape** | Animated 2-D canvas with budget/temperature sliders |
| **Compatibility matrix** | Pairwise ε_ij frustration table |
| **Budget calculator** | Theorem 6.1 N_max — no API key needed |
| **Historical Heatmap** | η_i accumulation across a reasoning trace (Addendum V) |
| **Providers** | API key configuration, per-provider status and model counts |

---

## HTTP API (localhost:7771)

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Browser UI (index.html) |
| `GET` | `/models` | Model catalogue using env-var keys |
| `POST` | `/models` | Refresh catalogue with caller-supplied keys |
| `GET` | `/env` | Which env vars are set (values redacted) |
| `POST` | `/status` | Provider connectivity health check |
| `POST` | `/setenv` | Persist a key to `.env` |
| `POST` | `/` | MPC actions (`compile`, `compile_sequence`, `read_claims`, `budget_estimate`, `free_energy_surface`, `ground_state`) |

---

## Testing

```bash
# Arithmetic tests (no API key required)
pytest tests/ -v

# Live API tests — any provider combination
ANTHROPIC_API_KEY=sk-ant-…  pytest tests/ -v
GOOGLE_API_KEY=AIza…        pytest tests/ -v
OPENAI_API_KEY=sk-…         pytest tests/ -v
KIMI_API_KEY=sk-…           pytest tests/ -v
```

---

## Architecture

```
mpc_core/
  providers.py     NEW v0.3 — ProviderID enum, per-provider key resolution,
                   dynamic model listing with TTL cache, all_models_catalogue()
  router.py        REWRITTEN — five-backend dispatch, retry logic, auth-error fast-fail
  compiler.py      compile() · read_claims() · budget_estimate() · compile_sequence()
  entity_ledger.py Cross-step entity registry (four-layer pipeline)
  thermodynamics.py QuTiP partition function · NetKet spin-glass · free_energy_surface()
  json_repair.py   Best-effort repair of truncated LLM JSON
  models.py        MPCResult dataclass hierarchy

mpc_server/
  server.py        REWRITTEN — FastMCP application + HTTP UI proxy (port 7771)
                   New endpoints: GET /env, POST /status, POST /models

static/
  index.html       REWRITTEN — provider selector tabs, per-provider key inputs,
                   dynamic model dropdowns, /status integration
```

---

## Roadmap

- **v0.1** ✓ Core compiler, MCP server, reference UI
- **v0.2** ✓ Multi-backend routing · QuTiP · NetKet · 3D Plotly · Historical heatmap
- **v0.3** ✓ FastMCP · Five providers · Dynamic model listing · Provider UI · Retry logic
- **v0.4** Spectral Laplacian extension of Theorem 6.1, community-aware N_max bounds
- **v0.5** Streaming analysis; differential η_i display per hypothesis per step

---

## License

MIT
