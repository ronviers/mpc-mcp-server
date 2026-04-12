"""
MPC data models — dataclasses mirroring the JSON schema returned by the compiler.
All energies in units of k_B T.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional

MPCPhase = Literal["c", "s", "k", "r"]

PHASE_NAMES = {"c": "Committed", "s": "Suspended", "k": "Conflict", "r": "Reset"}
PHASE_DESCRIPTIONS = {
    "c": "Deep potential minimum. Low holding cost, high revision cost.",
    "s": "Shallow minimum. Active maintenance required against collapse.",
    "k": "No satisfying configuration. Elevated cost until resolved.",
    "r": "Zero constraint potential. Maximally entropic prior.",
}

@dataclass
class Hypothesis:
    id: str
    text: str
    phase: MPCPhase
    barrier_height: float
    holding_cost: float
    linguistic_register: str
    rationale: str

@dataclass
class CompatibilityEntry:
    hi: str
    hj: str
    epsilon: float
    compatible: bool
    joint_phase: MPCPhase

@dataclass
class EnergyModel:
    E_star: float
    E_c: float
    E_s: float
    total_load: float
    budget_utilization: float
    dominant_phase: MPCPhase
    partition_function: float = 1.0
    free_energy: float = 0.0
    entropy: float = 0.0
    thermal_energy: float = 0.0
    thermo_backend: str = "none"
    matrix_mode: str = "dense"

@dataclass
class GroundState:
    energy: float
    config: list[int]
    stable_ids: list[str]
    backend: str

@dataclass
class BudgetEstimate:
    N: int
    d_avg: float
    epsilon_min: float
    alpha: float
    E_star: float
    N_max: float
    interpretation: str
    margin: float

@dataclass
class MPCResult:
    source_text: str
    hypotheses: list[Hypothesis]
    compatibility_matrix: list[CompatibilityEntry]
    energy_model: EnergyModel
    budget: BudgetEstimate
    k_states: list[str]
    analytical_summary: str
    ground_state: Optional[GroundState] = None
    model_used: str = ""

    def to_dict(self) -> dict:
        gs = None
        if self.ground_state:
            gs = {
                "energy": self.ground_state.energy,
                "config": self.ground_state.config,
                "stable_ids": self.ground_state.stable_ids,
                "backend": self.ground_state.backend,
            }
        return {
            "hypotheses": [
                {
                    "id": h.id, "text": h.text, "phase": h.phase,
                    "phase_name": PHASE_NAMES[h.phase],
                    "barrier_height": h.barrier_height,
                    "holding_cost": h.holding_cost,
                    "linguistic_register": h.linguistic_register,
                    "rationale": h.rationale,
                }
                for h in self.hypotheses
            ],
            "compatibility_matrix": [
                {
                    "hi": e.hi, "hj": e.hj, "epsilon": e.epsilon,
                    "compatible": e.compatible, "joint_phase": e.joint_phase,
                    "joint_phase_name": PHASE_NAMES[e.joint_phase],
                }
                for e in self.compatibility_matrix
            ],
            "energy_model": {
                "E_star": self.energy_model.E_star,
                "E_c": self.energy_model.E_c,
                "E_s": self.energy_model.E_s,
                "total_load": self.energy_model.total_load,
                "budget_utilization": self.energy_model.budget_utilization,
                "dominant_phase": self.energy_model.dominant_phase,
                "partition_function": self.energy_model.partition_function,
                "free_energy": self.energy_model.free_energy,
                "entropy": self.energy_model.entropy,
                "thermal_energy": self.energy_model.thermal_energy,
                "thermo_backend": self.energy_model.thermo_backend,
                "matrix_mode": self.energy_model.matrix_mode,
            },
            "budget": {
                "N": self.budget.N,
                "d_avg": self.budget.d_avg,
                "epsilon_min": self.budget.epsilon_min,
                "alpha": self.budget.alpha,
                "E_star": self.budget.E_star,
                "N_max": self.budget.N_max if self.budget.N_max != float("inf") else None,
                "interpretation": self.budget.interpretation,
                "margin": self.budget.margin if self.budget.margin != float("inf") else None,
            },
            "k_states": self.k_states,
            "analytical_summary": self.analytical_summary,
            "ground_state": gs,
            "model_used": self.model_used,
        }

@dataclass
class SequenceStep:
    step_index: int
    text: str
    result: MPCResult
    eta_matrix: dict[str, float]
    id_map: dict[str, str] = field(default_factory=dict)
    reconciliation_stats: dict[str, int] = field(default_factory=dict)

@dataclass
class SequenceResult:
    steps: list[SequenceStep]
    ledger: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "steps": [
                {
                    "step_index": s.step_index,
                    "text": s.text,
                    "result": s.result.to_dict(),
                    "eta_matrix": s.eta_matrix,
                    "id_map": s.id_map,
                    "reconciliation_stats": s.reconciliation_stats,
                }
                for s in self.steps
            ],
            "ledger": self.ledger,
        }

    def entity_timeline(self) -> dict[str, list[dict]]:
        timeline: dict[str, list[dict]] = {}
        for step in self.steps:
            for h in step.result.hypotheses:
                timeline.setdefault(h.id, []).append({
                    "step": step.step_index,
                    "phase": h.phase,
                    "eta": step.eta_matrix.get(h.id, 0.0),
                    "barrier": h.barrier_height,
                })
        return timeline
