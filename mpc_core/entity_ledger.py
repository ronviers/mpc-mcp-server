"""
mpc_core/entity_ledger.py

Layer 1 — State Ledger:  Canonical entity registry persisted across compile() calls.
Layer 2 — Local Triage:  CPU-bound string similarity pre-screen before any API call.

Usage in compile_sequence():
    ledger = EntityLedger()
    for text in texts:
        matches = ledger.pretriage(text)
        prompt  = ledger.build_context_block()
        result  = call_model(system+prompt, user)
        result  = ledger.reconcile(result)
        ledger.commit(result)
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

FUZZY_MATCH_THRESHOLD = 0.72
HARD_MATCH_THRESHOLD  = 0.88


@dataclass
class CanonicalEntity:
    canonical_id: str
    text: str
    phase: str
    barrier_height: float
    first_step: int
    last_step: int
    aliases: list[str] = field(default_factory=list)
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self):
        self._fingerprint = _fingerprint(self.text)

    def similarity(self, other_text: str) -> float:
        fp = _fingerprint(other_text)
        scores = [_ratio(self._fingerprint, fp)]
        for alias in self.aliases:
            scores.append(_ratio(_fingerprint(alias), fp))
        return max(scores)

    def update(self, text: str, phase: str, barrier_height: float, step: int):
        self.last_step = step
        self.phase = phase
        self.barrier_height = barrier_height
        fp = _fingerprint(text)
        if fp != self._fingerprint and fp not in [_fingerprint(a) for a in self.aliases]:
            self.aliases.append(text)


class EntityLedger:
    def __init__(self):
        self._entities: dict[str, CanonicalEntity] = {}
        self._counter  = 0
        self._step     = -1

    def advance_step(self):
        self._step += 1

    @property
    def step(self) -> int:
        return self._step

    @property
    def size(self) -> int:
        return len(self._entities)

    def canonical_ids(self) -> list[str]:
        return sorted(self._entities.keys(), key=_id_sort_key)

    def get(self, canonical_id: str) -> Optional[CanonicalEntity]:
        return self._entities.get(canonical_id)

    def pretriage(self, text: str) -> list[tuple[str, float]]:
        fp   = _fingerprint(text)
        hits = []
        for eid, ent in self._entities.items():
            score = ent.similarity(text)
            if score >= FUZZY_MATCH_THRESHOLD:
                hits.append((eid, score))
        return sorted(hits, key=lambda x: -x[1])

    def build_context_block(self) -> str:
        if not self._entities:
            return ""
        lines = ["<entity_ledger>"]
        for eid in self.canonical_ids():
            ent = self._entities[eid]
            aliases_str = " | ".join(ent.aliases[:3]) if ent.aliases else ""
            alias_attr  = f' aliases="{_escape(aliases_str)}"' if aliases_str else ""
            lines.append(
                f'  <entity id="{eid}" phase="{ent.phase}"'
                f' barrier_height="{ent.barrier_height:.2f}"'
                f' first_step="{ent.first_step}"'
                f' last_step="{ent.last_step}"{alias_attr}>'
                f'{_escape(ent.text)}</entity>'
            )
        lines.append("</entity_ledger>")
        return "\n".join(lines)

    def build_valid_ids_enum(self) -> list[str]:
        return self.canonical_ids()

    def reconcile(self, raw_data: dict, step: int) -> dict:
        if not raw_data.get("hypotheses"):
            return raw_data

        llm_id_to_canonical: dict[str, str] = {}
        new_hypotheses = []

        for h in raw_data["hypotheses"]:
            llm_id   = h["id"]
            llm_text = h.get("text", "")
            hits     = self.pretriage(llm_text)

            if hits and hits[0][1] >= HARD_MATCH_THRESHOLD:
                canonical_id = hits[0][0]
                llm_id_to_canonical[llm_id] = canonical_id
                h["id"] = canonical_id
                h["_reconciled"]   = "hard_match"
                h["_match_score"]  = round(hits[0][1], 3)

            elif hits and hits[0][1] >= FUZZY_MATCH_THRESHOLD:
                canonical_id = hits[0][0]
                llm_id_to_canonical[llm_id] = canonical_id
                h["id"] = canonical_id
                h["_reconciled"]  = "soft_match"
                h["_match_score"] = round(hits[0][1], 3)

            elif llm_id in self._entities:
                canonical_id = llm_id
                llm_id_to_canonical[llm_id] = canonical_id
                h["id"] = canonical_id
                h["_reconciled"]  = "llm_match"
                h["_match_score"] = 0.0

            else:
                canonical_id = self._next_id()
                llm_id_to_canonical[llm_id] = canonical_id
                h["id"] = canonical_id
                h["_reconciled"]  = "new"
                h["_match_score"] = hits[0][1] if hits else 0.0
                self._entities[canonical_id] = CanonicalEntity(
                    canonical_id=canonical_id,
                    text=llm_text,
                    phase=h.get("phase", "s"),
                    barrier_height=h.get("barrier_height", 0.0),
                    first_step=step,
                    last_step=step,
                )

            new_hypotheses.append(h)

        raw_data["hypotheses"] = new_hypotheses

        new_compat = []
        for entry in raw_data.get("compatibility_matrix", []):
            hi = llm_id_to_canonical.get(entry["hi"], entry["hi"])
            hj = llm_id_to_canonical.get(entry["hj"], entry["hj"])
            if hi != hj:
                entry["hi"] = hi
                entry["hj"] = hj
                new_compat.append(entry)
        raw_data["compatibility_matrix"] = new_compat

        raw_data["k_states"] = [
            llm_id_to_canonical.get(kid, kid)
            for kid in raw_data.get("k_states", [])
        ]

        raw_data["_id_map"] = llm_id_to_canonical
        return raw_data

    def commit(self, raw_data: dict, step: int):
        for h in raw_data.get("hypotheses", []):
            eid  = h["id"]
            text = h.get("text", "")
            if eid in self._entities:
                self._entities[eid].update(
                    text=text,
                    phase=h.get("phase", "s"),
                    barrier_height=h.get("barrier_height", 0.0),
                    step=step,
                )
            else:
                self._entities[eid] = CanonicalEntity(
                    canonical_id=eid,
                    text=text,
                    phase=h.get("phase", "s"),
                    barrier_height=h.get("barrier_height", 0.0),
                    first_step=step,
                    last_step=step,
                )

    def to_dict(self) -> dict:
        return {
            eid: {
                "text":           e.text,
                "phase":          e.phase,
                "barrier_height": e.barrier_height,
                "first_step":     e.first_step,
                "last_step":      e.last_step,
                "aliases":        e.aliases,
            }
            for eid, e in self._entities.items()
        }

    def _next_id(self) -> str:
        self._counter += 1
        return f"H{self._counter}"


def _fingerprint(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())

def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

def _id_sort_key(eid: str) -> tuple:
    m = re.match(r"([A-Za-z]+)(\d+)", eid)
    if m:
        return (m.group(1), int(m.group(2)))
    return (eid, 0)
