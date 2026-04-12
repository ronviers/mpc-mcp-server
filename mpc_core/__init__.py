"""
mpc_core — Metastable Propositional Calculus core library.

Public API
----------
compile(text, api_key, *, model, …)            → MPCResult
read_claims(claims, api_key, …)                → list[dict]
budget_estimate(N, d_avg, …)                   → BudgetEstimate
compile_sequence(texts, api_key, …)            → SequenceResult
"""
from .compiler import compile, read_claims, budget_estimate, compile_sequence   # noqa: F401
from .entity_ledger import EntityLedger                                         # noqa: F401
from .models import MPCResult                                                   # noqa: F401
from .providers import (                                                        # noqa: F401
    DEFAULT_MODEL,
    ProviderID,
    all_models_catalogue,
    list_models,
    provider_for_model,
    resolve_api_key,
)
from .router import call_model                                                  # noqa: F401
from .thermodynamics import (                                                   # noqa: F401
    compute_thermodynamic_quantities,
    find_ground_state,
    free_energy_surface,
)
