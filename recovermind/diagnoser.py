"""Component D - Failure Diagnoser.

Turns an alarm into a typed root cause (class, locus) with a calibrated confidence,
or abstains. Three mechanisms, exactly as specified in the framework document:

  1. multi-hypothesis scoring    - evidence x memory prior, softmaxed
  2. confidence as the margin    - a narrow win counts as uncertain
  3. counterfactual probe        - a read-only experiment chosen by information gain

The evidence model is the taxonomy's own signal signatures, so the Diagnoser is
never told the answer: it compares what the Monitor observed against what each
subclass is *expected* to look like.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from env import tools
from recovermind.taxonomy import SUBCLASSES, family
from recovermind.types import AgentState, Diagnosis, Signals, Step

TAU_C = 0.12          # abstention threshold on the posterior margin
BETA = 4.0            # how strongly current evidence overrides the memory prior
N_PROBE = 2


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    m = max(scores.values())
    exp = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(exp.values())
    return {k: v / z for k, v in exp.items()}


class FailureDiagnoser:
    def __init__(self, env: tools.Environment, memory=None):
        self.env = env
        self.memory = memory
        self.probe_log: List[str] = []

    # ------------------------------------------------------------- evidence
    @staticmethod
    def _idf(signal: str) -> float:
        """Specificity of a signal: rare signals are strong evidence, common ones weak."""
        n = sum(1 for c in SUBCLASSES if signal in SUBCLASSES[c].signals)
        return math.log(len(SUBCLASSES) / max(1, n))

    @classmethod
    def _evidence(cls, observed: frozenset, code: str) -> float:
        """Specificity-weighted overlap between what fired and what this subclass predicts.

        A hypothesis is good when it explains the *rare* signals that fired and leaves
        little of the observed evidence unaccounted for.
        """
        expected = SUBCLASSES[code].signals
        if not expected:
            return 0.0
        w = cls._idf
        exp_mass = sum(w(s) for s in expected) or 1.0
        obs_mass = sum(w(s) for s in observed) or 1.0
        hit = sum(w(s) for s in observed & expected) / exp_mass
        noise = sum(w(s) for s in expected - observed) / exp_mass
        extra = sum(w(s) for s in observed - expected) / obs_mass
        return max(0.0, hit - 0.35 * noise - 0.90 * extra)

    def _candidates(self, observed: frozenset) -> List[str]:
        """Restrict hypotheses to subclasses with at least one matching signal."""
        cands = [c for c in SUBCLASSES if observed & SUBCLASSES[c].signals]
        return cands or list(SUBCLASSES)

    # ------------------------------------------------------------- probes
    def _probe(self, code: str, st: AgentState, step: Step) -> Optional[bool]:
        """Evaluate a subclass's discriminating predicate with a read-only call.

        Returns True if the predicate *supports* the hypothesis, False if it refutes
        it, None if it cannot be evaluated here.
        """
        pred = SUBCLASSES[code].probe
        paths = [v for k, v in step.args.items() if k == "path" and isinstance(v, str)]
        try:
            if pred == "tool_in_registry":
                r = tools.call(self.env, "lookup_schema", {"name": step.tool})
                self.probe_log.append("lookup_schema(%s) -> exists=%s" % (step.tool, r["exists"]))
                return not r["exists"]
            if pred == "referent_resolves" and paths:
                r = tools.call(self.env, "file_exists", {"path": paths[0]})
                self.probe_log.append("file_exists(%s) -> %s" % (paths[0], r["exists"]))
                return not r["exists"]
            if pred == "args_validate" and step.tool in tools.REGISTRY:
                bad = tools.validate_args(step.tool, step.args)
                self.probe_log.append("validate_args(%s) -> %s" % (step.tool, bad or "ok"))
                return bad is not None
            if pred == "constraints_present":
                from env.agent import REQUIRED_CONSTRAINTS
                miss = [c for c in REQUIRED_CONSTRAINTS if c not in st.constraints]
                self.probe_log.append("constraints_present -> missing=%d" % len(miss))
                return bool(miss)
            if pred == "fact_has_support":
                bad = [k for k in st.facts if st.fact_support.get(k) is None]
                self.probe_log.append("fact_has_support -> ungrounded=%s" % bad)
                return bool(bad)
        except tools.ToolError:
            return None
        return None

    # ------------------------------------------------------------- diagnose
    def diagnose(self, st: AgentState, step: Step, sig: Signals) -> Optional[Diagnosis]:
        observed = frozenset(sig.fired())
        if not observed:
            return None

        cands = self._candidates(observed)
        prior = self.memory.diagnosis_prior(observed, cands) if self.memory else {}
        scores = {c: BETA * self._evidence(observed, c) + math.log(prior.get(c, 1.0 / len(cands)))
                  for c in cands}
        post = _softmax(scores)

        probes = 0
        while probes < N_PROBE:
            ranked = sorted(post.items(), key=lambda kv: -kv[1])
            if len(ranked) < 2 or ranked[0][1] - ranked[1][1] >= TAU_C:
                break
            # probe the leading hypothesis: cheapest way to break the tie
            target = ranked[0][0]
            verdict = self._probe(target, st, step)
            if verdict is None:
                target = ranked[1][0]
                verdict = self._probe(target, st, step)
            if verdict is None:
                break
            factor = 3.0 if verdict else 0.2
            post[target] *= factor
            z = sum(post.values())
            post = {k: v / z for k, v in post.items()}
            probes += 1

        ranked = sorted(post.items(), key=lambda kv: -kv[1])
        margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
        if margin < TAU_C * 0.5:
            return None                                   # abstain, do not guess

        best = ranked[0][0]
        locus = "step:%d/tool:%s" % (step.t, step.tool)
        return Diagnosis(subclass=best, locus=locus, confidence=round(margin, 3),
                         ranked=[(c, round(p, 3)) for c, p in ranked[:4]],
                         evidence=sorted(observed), probes_used=probes)
