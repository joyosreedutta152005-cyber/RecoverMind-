"""Component R - Recovery Memory.

Case-based reasoning with one restriction that does all the work: a case is admitted
only when the Verifier actually evaluated the post-condition. Unverified attempts are
never stored, so the policy cannot learn from its own unchecked self-assessment.

Provides
  * diagnosis priors  pi_R(h | signature)     -> read by the Diagnoser
  * strategy values   Q(s | subclass)         -> read by the Planner
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, List, Optional

from recovermind.types import Case

HALF_LIFE = 400.0          # cases decay with age, in episodes
PRIOR_SMOOTHING = 0.5      # how far the diagnosis prior is pulled toward uniform
SIM_FLOOR = 0.75           # a past case must be a near-exact signature match to vote
EXPLORE_K = 0.45           # UCB exploration coefficient


class RecoveryMemory:
    def __init__(self, path: Optional[str] = None):
        self.cases: List[Case] = []
        self.clock = 0
        self.path = path
        if path and os.path.exists(path):
            self.load(path)

    # ------------------------------------------------------------- write
    def record(self, signature: str, subclass: str, strategy: str,
               outcome: int, context: str, step_cost: float) -> None:
        """Only ever called by the Verifier, and only for evaluated post-conditions."""
        self.clock += 1
        self.cases.append(Case(signature, subclass, strategy, outcome,
                               context, step_cost, self.clock))

    # ------------------------------------------------------------- retrieval
    def _weight(self, c: Case) -> float:
        age = self.clock - c.inserted_at
        return math.exp(-age / HALF_LIFE)

    def diagnosis_prior(self, observed: Iterable[str], candidates: List[str]) -> Dict[str, float]:
        """pi_R(h | signature): similarity- and recency-weighted vote over past cases."""
        obs = set(observed)
        votes: Dict[str, float] = {c: 0.0 for c in candidates}
        for c in self.cases:
            past = set(c.signature.split("|"))
            if not past:
                continue
            sim = len(obs & past) / float(len(obs | past) or 1)
            # A partial overlap is not a match: many failure classes share a generic
            # signal such as tool_error, and letting those vote leaks probability mass
            # between unrelated subclasses and corrupts the diagnosis.
            if sim < SIM_FLOOR or c.subclass not in votes:
                continue
            votes[c.subclass] += sim * self._weight(c)
        total = sum(votes.values())
        uniform = 1.0 / len(candidates)
        if total <= 0:
            return {c: uniform for c in candidates}
        # Smooth halfway toward uniform. This bounds log(prior) to a narrow band, so
        # the prior can break a tie but cannot outvote the evidence term.
        return {k: PRIOR_SMOOTHING * uniform + (1.0 - PRIOR_SMOOTHING) * (v / total)
                for k, v in votes.items()}

    def value(self, strategy: str, subclass: str) -> "tuple[float, int]":
        """Q(s|d) and the trial count n(s,d), from verified cases only."""
        rel = [c for c in self.cases if c.subclass == subclass and c.strategy == strategy]
        if not rel:
            return 0.0, 0
        w = [self._weight(c) for c in rel]
        q = sum(wi * c.outcome for wi, c in zip(w, rel)) / sum(w)
        return q, len(rel)

    def trials_for(self, subclass: str) -> int:
        return sum(1 for c in self.cases if c.subclass == subclass)

    def ucb_bonus(self, strategy: str, subclass: str) -> float:
        n = self.value(strategy, subclass)[1]
        N = max(self.trials_for(subclass), 2)
        return EXPLORE_K * math.sqrt(math.log(N) / (1 + n))

    # ------------------------------------------------------------- housekeeping
    def consolidate(self, floor: float = 0.02) -> int:
        before = len(self.cases)
        self.cases = [c for c in self.cases if self._weight(c) >= floor]
        return before - len(self.cases)

    def stats(self) -> Dict[str, object]:
        ok = sum(c.outcome for c in self.cases)
        return {"cases": len(self.cases), "verified_successes": ok,
                "verified_failures": len(self.cases) - ok,
                "subclasses_seen": len({c.subclass for c in self.cases})}

    # ------------------------------------------------------------- persistence
    def save(self, path: Optional[str] = None) -> None:
        path = path or self.path
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf8") as fh:
            json.dump({"clock": self.clock, "cases": [c.__dict__ for c in self.cases]},
                      fh, indent=1)

    def load(self, path: str) -> None:
        with open(path, encoding="utf8") as fh:
            blob = json.load(fh)
        self.clock = blob.get("clock", 0)
        self.cases = [Case(**c) for c in blob.get("cases", [])]
