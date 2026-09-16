"""Component P - Adaptive Recovery Planner.

Ranks the closed operator library by expected utility

    U(s | d, x) = Qhat(s|d,x) - lambda_c * cost(s) - lambda_r * rev(s) * (1 - confidence)

The last term is the safety coupling: the penalty on irreversible operators is scaled
by *diagnostic uncertainty*, so a confident diagnosis may justify an environment-level
repair while an uncertain one is pushed toward cheap reversible edits.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from recovermind.memory import RecoveryMemory
from recovermind.strategies import LIBRARY, REPAIR_OPERATORS
from recovermind.taxonomy import SUBCLASSES
from recovermind.types import Diagnosis

LAMBDA_C = 0.35     # weight on cost
LAMBDA_R = 0.55     # weight on irreversibility


class RecoveryPlanner:
    def __init__(self, memory: Optional[RecoveryMemory] = None,
                 r_max: float = 1.0, use_memory: bool = True):
        self.memory = memory
        self.r_max = r_max              # deployment irreversibility ceiling
        self.use_memory = use_memory

    # ------------------------------------------------------------- priors
    @staticmethod
    def taxonomy_prior(strategy: str, subclass: str) -> float:
        """Cold-start value: what the taxonomy says should work for this class."""
        sc = SUBCLASSES[subclass]
        if strategy == sc.canonical:
            return 0.80
        if strategy == sc.secondary:
            return 0.45
        fam_ops = {SUBCLASSES[c].canonical for c in SUBCLASSES
                   if c.split(".")[0] == subclass.split(".")[0]}
        return 0.15 if strategy in fam_ops else 0.05

    # ------------------------------------------------------------- ranking
    def rank(self, d: Diagnosis, budget: float,
             tried: Optional[List[str]] = None) -> List[Tuple[str, float, str]]:
        tried = tried or []
        out: List[Tuple[str, float, str]] = []

        for sid in REPAIR_OPERATORS:
            op = LIBRARY[sid]
            if sid in tried:
                continue
            if op.reversibility > self.r_max:
                continue                                  # forbidden in this deployment
            if op.cost > budget:
                continue                                  # unaffordable

            prior = self.taxonomy_prior(sid, d.subclass)
            why = "taxonomy prior"
            q = prior
            if self.use_memory and self.memory is not None:
                mq, n = self.memory.value(sid, d.subclass)
                if n > 0:
                    q = (n * mq + 2.0 * prior) / (n + 2.0)   # shrink toward the prior
                    why = "memory n=%d q=%.2f" % (n, mq)
                q += self.memory.ucb_bonus(sid, d.subclass)

            u = q - LAMBDA_C * op.cost - LAMBDA_R * op.reversibility * (1.0 - d.confidence)
            out.append((sid, round(u, 3), why))

        out.sort(key=lambda x: -x[1])
        if not out:
            return [("S9", 0.0, "no admissible operator within budget")]
        return out

    def explain(self, d: Diagnosis, ranked: List[Tuple[str, float, str]]) -> List[str]:
        lines = []
        for sid, u, why in ranked[:4]:
            op = LIBRARY[sid]
            lines.append("      %-3s U=%+.3f  cost=%.2f rev=%.1f  %-38s (%s)"
                         % (sid, u, op.cost, op.reversibility, op.name, why))
        return lines
