"""
mpc_core/compiler.py

Entry points:
  compile(text, api_key, *, model, …)         → MPCResult
  read_claims(claims, api_key, …)             → list[dict]
  budget_estimate(N, d_avg, …)               → BudgetEstimate
  compile_sequence(texts, api_key, …)         → SequenceResult

All energies in units of k_B T.  Thresholds: E_c = 1.0, E_s = 3.0.

Two-pass matrix routing
-----------------------
Pass 1 (cheap): extract hypothesis IDs and count only.
Pass 2 (full):  full MPC analysis. Matrix instruction chosen by Python:
                  N < 10  → dense  (all pairs)
                  N ≥ 10  → sparse (omit pairs where ε = 0.0)
"""
from __future__ import annotations

import json
import math
import re
from typing import Optional

from .router import call_model
from .providers import DEFAULT_MODEL
from .json_repair import loads_or_repair
from .thermodynamics import compute_thermodynamic_quantities, find_ground_state
from .models import (
    BudgetEstimate, CompatibilityEntry, EnergyModel, GroundState,
    Hypothesis, MPCResult, PHASE_NAMES,
)

DEFAULT_E_C    = 1.0
DEFAULT_E_S    = 3.0
DEFAULT_E_STAR = 20.0
DEFAULT_ALPHA  = 1.0
DEFAULT_TEMP   = 1.0
_SPARSE_THRESHOLD = 10

_DENSE_MATRIX_RULE = (
    "- compatibility_matrix: include ALL pairs (i < j only, no self-pairs).\n"
    "- epsilon = 0.0 when propositions are orthogonal (no shared variables)."
)
_SPARSE_MATRIX_RULE = (
    "- compatibility_matrix: include ONLY pairs where epsilon > 0.0 "
    "(i < j only, no self-pairs).  Omit orthogonal pairs entirely — "
    "the Python builder initialises all missing pairs to 0.0 automatically."
)

_PASS1_SYSTEM = """\
You are an MPC hypothesis extractor.  Your only task is to identify the
distinct propositional units in the text and return them as a JSON object.
No compatibility matrix, no energy model, no summary — hypotheses only.

Rules:
- Extract 3-12 distinct propositional units.  Merge near-duplicates.
- Assign sequential ids: H1, H2, H3, ...
- Return ONLY valid JSON — no markdown fences, no preamble.

OUTPUT SCHEMA:
{"hypotheses": [{"id": "H1", "text": "<propositional unit>"}]}
"""

_SYSTEM_PROMPT = """\
You are a formal logic and thermodynamics analyst implementing Metastable
Propositional Calculus (MPC).  Your task is to analyse a passage of text and
return a structured JSON object — nothing else, no markdown fences, no preamble.

MPC truth values (all energies in units of k_B T):
  c (Committed)  — deep potential minimum, E < E_c.  Hard constraints:
                   "must", "always", "cannot", "strictly", axioms, definitions.
  s (Suspended)  — shallow minimum, E_c <= E <= E_s.  Soft constraints:
                   "might", "tends to", "probably", open questions.
  k (Conflict)   — no satisfying configuration, E > E_s.  Mutually
                   incompatible commitments held simultaneously.
  r (Reset)      — zero potential, identity element.  Tautologies,
                   trivially true or vacuous statements.

Thresholds: E_c={E_c}, E_s={E_s}, E_star={E_star} (budget), alpha={alpha}.

The following hypothesis ids have already been extracted in a prior pass.
Use EXACTLY these ids — do not renumber or rename them:
{hypothesis_seed}

OUTPUT SCHEMA (strict JSON, no extra keys):
{{
  "hypotheses": [
    {{
      "id": "H1",
      "text": "<propositional unit verbatim or closely paraphrased>",
      "phase": "c"|"s"|"k"|"r",
      "barrier_height": <float, delta-E in k_B T units>,
      "holding_cost": <float, P*tau estimate in k_B T units>,
      "linguistic_register": "hard"|"soft"|"contradictory"|"neutral",
      "rationale": "<one sentence>"
    }},
    ...
  ],
  "compatibility_matrix": [
    {{
      "hi": "H1",
      "hj": "H2",
      "epsilon": <float, pairwise frustration in k_B T>,
      "compatible": true|false,
      "joint_phase": "c"|"s"|"k"|"r"
    }},
    ...
  ],
  "energy_model": {{
    "total_load": <float>,
    "budget_utilization": <float 0-1>,
    "dominant_phase": "c"|"s"|"k"|"r"
  }},
  "k_states": ["H3", ...],
  "analytical_summary": "<2-3 sentences on the thermodynamic structure>"
}}

Rules:
{matrix_rule}
- epsilon > 0 when joint satisfaction is frustrated; scale 0-10.
- compatible = true iff epsilon < E_s.
- joint_phase uses the Commitment operator C(Hi, Hj):
    c if joint energy < E_c, s if <= E_s, k if > E_s.
- holding_cost for phase s: barrier_height / 10 as proxy.
- holding_cost for phase c or r: 0.0.
- holding_cost for phase k: barrier_height (sustained cost).
- total_load = sum of all holding_costs.
- budget_utilization = total_load / E_star (clamp to 1.0 max).
- k_states = list of hypothesis IDs whose phase is "k".
- Return ONLY valid JSON.
"""

_SEQUENCE_ADDENDUM = """
{ledger_block}

<instructions>
  You are analysing step {step_num} of a multi-step reasoning trace.

  ENTITY IDENTITY RULES (strictly enforced):
  1. For every hypothesis you extract, first check whether it substantially
     matches an entity already recorded in <entity_ledger>.
  2. If a match exists, you MUST reuse the existing id exactly as shown.
  3. Only mint a new id (continuing from H{next_id}) for genuinely new propositions.
  4. Valid ids: [{valid_ids}]
  5. compatibility_matrix must use only ids from this response or valid_ids above.

  <planning>
  Before writing JSON, silently:
    a) List candidate propositions from the text.
    b) Check each against ledger entities.
    c) Assign: REUSE or MINT.
    d) Produce JSON.
  </planning>
</instructions>
"""

_PASS1_SEQUENCE_ADDENDUM = """
{ledger_block}

<instructions>
  Reuse existing ids from <entity_ledger> for known concepts.
  Mint new ids (starting from H{next_id}) only for genuinely new propositions.
  Return ONLY the hypotheses array.
</instructions>
"""


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$",          "", text).strip()
    return text


def _pass1_extract(
    text: str,
    model: str,
    api_key: str,
    system_addendum: str = "",
) -> tuple[int, list[dict]]:
    system = _PASS1_SYSTEM + ("\n" + system_addendum if system_addendum else "")
    user   = f"Extract hypotheses from the following text:\n\n{text}"
    try:
        raw  = call_model(system, user, model=model, api_key=api_key, max_tokens=1024)
        raw  = _strip_fences(raw)
        data = loads_or_repair(raw)
        hyps = data.get("hypotheses", [])
        return len(hyps), hyps
    except Exception:
        return 0, []


def _hypothesis_seed(hyps: list[dict]) -> str:
    if not hyps:
        return "(none — assign ids H1, H2, ... sequentially)"
    return "\n".join(f'  {h["id"]}: {h["text"]}' for h in hyps)


def compile(
    text: str,
    api_key: str = "",
    *,
    E_c:    float = DEFAULT_E_C,
    E_s:    float = DEFAULT_E_S,
    E_star: float = DEFAULT_E_STAR,
    alpha:  float = DEFAULT_ALPHA,
    temperature: float = DEFAULT_TEMP,
    model:  str   = DEFAULT_MODEL,
    solve_ground_state: bool = True,
) -> MPCResult:
    """Full two-pass MPC analysis of *text*."""
    n, hyps = _pass1_extract(text, model, api_key)
    matrix_rule = _SPARSE_MATRIX_RULE if n >= _SPARSE_THRESHOLD else _DENSE_MATRIX_RULE
    matrix_mode = "sparse" if n >= _SPARSE_THRESHOLD else "dense"

    system = _SYSTEM_PROMPT.format(
        E_c=E_c, E_s=E_s, E_star=E_star, alpha=alpha,
        matrix_rule=matrix_rule,
        hypothesis_seed=_hypothesis_seed(hyps),
    )
    user = f"Analyse the following text using MPC:\n\n{text}"
    raw  = call_model(system, user, model=model, api_key=api_key, max_tokens=8192)
    raw  = _strip_fences(raw)
    data = loads_or_repair(raw)
    data["_matrix_mode"] = matrix_mode

    return _build_result(text, data, E_c, E_s, E_star, alpha, temperature, model, solve_ground_state)


def read_claims(
    claims: list[str],
    api_key: str = "",
    *,
    E_c:   float = DEFAULT_E_C,
    E_s:   float = DEFAULT_E_S,
    model: str   = DEFAULT_MODEL,
) -> list[dict]:
    """Assign an MPC phase to each claim independently."""
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    system = (
        "You are an MPC phase classifier.  For each numbered claim return "
        "ONLY a JSON array — no fences, no prose — where each element is:\n"
        '{"index": <int>, "phase": "c"|"s"|"k"|"r", "barrier_height": <float>, '
        '"linguistic_register": "hard"|"soft"|"contradictory"|"neutral", '
        '"rationale": "<one sentence>"}\n\n'
        f"Thresholds: E_c={E_c}, E_s={E_s}.  "
        "c = hard committed fact/axiom.  "
        "s = hedged/uncertain/open hypothesis.  "
        "k = claim that directly contradicts another claim in the list.  "
        "r = tautology/vacuous/trivially true."
    )
    user = f"Classify these claims:\n{numbered}"
    raw  = call_model(system, user, model=model, api_key=api_key, max_tokens=4096)
    raw  = _strip_fences(raw)
    try:
        phases = json.loads(raw)
        if isinstance(phases, dict):
            phases = next((v for v in phases.values() if isinstance(v, list)), [])
    except json.JSONDecodeError:
        try:
            recovered = loads_or_repair(raw)
            phases = next((v for v in recovered.values() if isinstance(v, list)), [])
        except Exception:
            phases = []

    results = []
    for i, claim in enumerate(claims):
        match = next((p for p in phases if p.get("index") == i + 1), None)
        if match:
            results.append({
                "claim": claim,
                "phase": match["phase"],
                "phase_name": PHASE_NAMES[match["phase"]],
                "barrier_height": match.get("barrier_height", 0.0),
                "linguistic_register": match.get("linguistic_register", "neutral"),
                "rationale": match.get("rationale", ""),
            })
        else:
            results.append({
                "claim": claim, "phase": "r", "phase_name": "Reset",
                "barrier_height": 0.0, "linguistic_register": "neutral",
                "rationale": "No classification returned.",
            })
    return results


def budget_estimate(
    N:           int,
    d_avg:       float,
    epsilon_min: float,
    alpha:       float = DEFAULT_ALPHA,
    E_star:      float = DEFAULT_E_STAR,
) -> BudgetEstimate:
    """Compute Theorem 6.1 bound N_max = O(sqrt(2E* / alpha * eps_min * d_avg))."""
    if d_avg <= 0 or epsilon_min <= 0:
        N_max = float("inf")
        interpretation = (
            "With zero average degree or zero minimum frustration the bound "
            "is vacuous: all hypotheses are simultaneously sustainable."
        )
    else:
        N_max  = math.sqrt(2 * E_star / (alpha * epsilon_min * d_avg))
        margin = N_max - N
        headroom = (
            "well within"   if margin > 5  else
            "approaching"   if margin > 0  else
            "exceeding"
        )
        interpretation = (
            f"With N={N} hypotheses, average degree d_avg={d_avg:.2f}, "
            f"minimum frustration eps_min={epsilon_min:.2f} k_B T, "
            f"budget E*={E_star:.1f} k_B T, and substrate efficiency alpha={alpha:.2f}, "
            f"Theorem 6.1 gives N_max ≈ {N_max:.1f}.  "
            f"The system is {headroom} its thermodynamic capacity "
            f"(margin = {margin:.1f} hypotheses)."
        )

    margin = (N_max - N) if N_max != float("inf") else float("inf")
    return BudgetEstimate(
        N=N, d_avg=d_avg, epsilon_min=epsilon_min,
        alpha=alpha, E_star=E_star, N_max=N_max,
        interpretation=interpretation, margin=margin,
    )


def compile_sequence(
    texts:              list[str],
    api_key:            str   = "",
    *,
    E_c:                float = DEFAULT_E_C,
    E_s:                float = DEFAULT_E_S,
    E_star:             float = DEFAULT_E_STAR,
    alpha:              float = DEFAULT_ALPHA,
    temperature:        float = DEFAULT_TEMP,
    model:              str   = DEFAULT_MODEL,
    solve_ground_state: bool  = True,
) -> "SequenceResult":
    """Multi-step trace analysis with cross-step entity consistency."""
    from .models import SequenceStep, SequenceResult
    from .entity_ledger import EntityLedger

    ledger           = EntityLedger()
    steps            = []
    accumulated_eta: dict[str, float] = {}

    for i, text in enumerate(texts):
        ledger.advance_step()
        ledger_block = ledger.build_context_block()
        valid_ids    = ledger.canonical_ids()
        next_id_num  = ledger.size + 1

        p1_addendum = ""
        if ledger_block:
            p1_addendum = _PASS1_SEQUENCE_ADDENDUM.format(
                ledger_block=ledger_block, next_id=next_id_num,
            )
        n, p1_hyps = _pass1_extract(text, model, api_key, system_addendum=p1_addendum)

        n_total     = ledger.size + max(n - ledger.size, 0)
        matrix_rule = (_SPARSE_MATRIX_RULE if n_total >= _SPARSE_THRESHOLD else _DENSE_MATRIX_RULE)
        matrix_mode = "sparse" if n_total >= _SPARSE_THRESHOLD else "dense"

        system = _SYSTEM_PROMPT.format(
            E_c=E_c, E_s=E_s, E_star=E_star, alpha=alpha,
            matrix_rule=matrix_rule,
            hypothesis_seed=_hypothesis_seed(p1_hyps),
        )
        if ledger_block:
            system += "\n" + _SEQUENCE_ADDENDUM.format(
                ledger_block=ledger_block,
                step_num=i + 1,
                next_id=next_id_num,
                valid_ids=", ".join(valid_ids) if valid_ids else "none yet",
            )
        else:
            system += (
                "\n<instructions>"
                "\n  This is step 1 of a multi-step trace. Assign ids H1, H2, ... sequentially."
                "\n</instructions>"
            )

        user = f"Analyse the following text using MPC:\n\n{text}"
        raw  = call_model(system, user, model=model, api_key=api_key, max_tokens=8192)
        raw  = _strip_fences(raw)
        data = loads_or_repair(raw)
        data["_matrix_mode"] = matrix_mode
        data = ledger.reconcile(data, step=i)
        ledger.commit(data, step=i)

        result = _build_result(
            text, data, E_c, E_s, E_star, alpha, temperature, model, solve_ground_state,
        )

        current_eta: dict[str, float] = {}
        for h in result.hypotheses:
            prev = accumulated_eta.get(h.id, 0.0)
            new_eta = prev + h.holding_cost
            current_eta[h.id]     = new_eta
            accumulated_eta[h.id] = new_eta

        steps.append(SequenceStep(
            step_index=i,
            text=text,
            result=result,
            eta_matrix=current_eta.copy(),
            id_map=data.get("_id_map", {}),
            reconciliation_stats=_reconciliation_stats(data),
        ))

    return SequenceResult(steps=steps, ledger=ledger.to_dict())


def _build_result(
    source_text: str,
    data: dict,
    E_c: float, E_s: float, E_star: float, alpha: float, temperature: float,
    model: str, solve_gs: bool,
) -> MPCResult:
    hypotheses = [
        Hypothesis(
            id=h["id"], text=h["text"], phase=h["phase"],
            barrier_height=h.get("barrier_height", 0.0),
            holding_cost=h.get("holding_cost", 0.0),
            linguistic_register=h.get("linguistic_register", "neutral"),
            rationale=h.get("rationale", ""),
        )
        for h in data["hypotheses"]
    ]
    compat = [
        CompatibilityEntry(
            hi=e["hi"], hj=e["hj"],
            epsilon=e.get("epsilon", 0.0),
            compatible=e.get("compatible", True),
            joint_phase=e.get("joint_phase", "r"),
        )
        for e in data.get("compatibility_matrix", [])
    ]
    N   = len(hypotheses)
    ids = [h.id for h in hypotheses]
    idx = {hid: i for i, hid in enumerate(ids)}
    eps_matrix = [[0.0] * N for _ in range(N)]
    for e in compat:
        if e.hi in idx and e.hj in idx:
            i, j = idx[e.hi], idx[e.hj]
            eps_matrix[i][j] = e.epsilon
            eps_matrix[j][i] = e.epsilon

    td = compute_thermodynamic_quantities(eps_matrix, temperature=temperature)

    em_raw      = data.get("energy_model", {})
    total_load  = em_raw.get("total_load", sum(h.holding_cost for h in hypotheses))
    budget_util = min(total_load / E_star, 1.0) if E_star > 0 else 1.0

    from collections import Counter
    phase_counts = Counter(h.phase for h in hypotheses)
    dominant     = phase_counts.most_common(1)[0][0] if phase_counts else "r"

    energy_model = EnergyModel(
        E_star=E_star, E_c=E_c, E_s=E_s,
        total_load=total_load,
        budget_utilization=em_raw.get("budget_utilization", budget_util),
        dominant_phase=em_raw.get("dominant_phase", dominant),
        partition_function=td["Z"],
        free_energy=round(td["F"], 4),
        entropy=round(td["S"], 4),
        thermal_energy=round(td["E_avg"], 4),
        thermo_backend=td["backend"],
        matrix_mode=data.get("_matrix_mode", "dense"),
    )

    k_ids  = data.get("k_states", [h.id for h in hypotheses if h.phase == "k"])
    edges  = [e for e in compat if e.epsilon > 0]
    d_avg  = (2 * len(edges) / N) if N > 0 else 0.0
    eps_min = min((e.epsilon for e in edges), default=0.0)
    bud    = budget_estimate(N, d_avg, eps_min, alpha, E_star)

    gs_result: Optional[GroundState] = None
    if solve_gs and N > 0:
        gs_raw     = find_ground_state(eps_matrix)
        stable_ids = [ids[i] for i in gs_raw["stable_ids"] if i < len(ids)]
        gs_result  = GroundState(
            energy=round(gs_raw["energy"], 4),
            config=gs_raw["config"],
            stable_ids=stable_ids,
            backend=gs_raw["backend"],
        )

    return MPCResult(
        source_text=source_text,
        hypotheses=hypotheses,
        compatibility_matrix=compat,
        energy_model=energy_model,
        budget=bud,
        k_states=k_ids,
        analytical_summary=data.get("analytical_summary", ""),
        ground_state=gs_result,
        model_used=model,
    )


def _reconciliation_stats(data: dict) -> dict:
    stats = {"hard_match": 0, "soft_match": 0, "llm_match": 0, "new": 0}
    for h in data.get("hypotheses", []):
        key = h.get("_reconciled", "new")
        stats[key] = stats.get(key, 0) + 1
    return stats
