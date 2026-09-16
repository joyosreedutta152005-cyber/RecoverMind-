"""Component M - Execution Monitor.

Runs on every step. Never calls the host model. Computes four signal families and
fuses them into a single anomaly score; also maintains the rollback checkpoint.

Every invariant below is a real computation over the trace, the agent state and the
environment. Nothing is told to the Monitor by the fault injector.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from env import tools
from env.agent import REQUIRED_CONSTRAINTS
from recovermind.types import AgentState, Signals, Step

TOKEN_LIMIT = 60000
WINDOW = 4
W_I, W_V, W_G, W_L, BIAS = 0.55, 0.28, 0.22, 0.45, 0.30
TAU_DET = 0.50


def _sigmoid(x: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-6.0 * (x - 0.5)))


class ExecutionMonitor:
    def __init__(self, env: tools.Environment):
        self.env = env
        self.history: List[Step] = []
        self.checkpoint: Optional[Tuple[AgentState, Dict[str, Any], int]] = None
        self._last_cursor = 0
        self._stalled_for = 0

    # ------------------------------------------------------------- invariants
    def _invariants(self, st: AgentState, step: Step) -> Dict[str, bool]:
        inv: Dict[str, bool] = {}

        # --- tool layer
        inv["tool_unknown"] = step.tool not in tools.REGISTRY and step.tool != "noop"
        inv["schema_invalid"] = bool(
            step.tool in tools.REGISTRY
            and tools.validate_args(step.tool, step.args) is not None)
        inv["tool_error"] = (not step.ok)
        inv["unguarded_write"] = step.tool in tools.DESTRUCTIVE

        # --- referents: a path argument that does not resolve
        paths = [v for k, v in step.args.items() if k == "path" and isinstance(v, str)]
        inv["referent_missing"] = any(
            p not in self.env.fs and step.tool in ("read_file", "delete_file")
            for p in paths)

        # --- retrieval relevance: a document was fetched that resolves but is the
        #     wrong kind. Evaluated over the window, because the consequence surfaces
        #     one or more steps after the retrieval itself.
        offtarget = False
        for past in (self.history[-WINDOW:] + [step]):
            if past.tool != "read_file" or not past.ok:
                continue
            pp = past.args.get("path")
            if isinstance(pp, str) and pp in self.env.fs and not pp.endswith(".csv"):
                offtarget = True
        inv["retrieval_offtarget"] = offtarget

        # --- context and memory
        inv["token_limit"] = st.tokens > TOKEN_LIMIT
        inv["constraint_missing"] = any(c not in st.constraints
                                        for c in REQUIRED_CONSTRAINTS)
        inv["stale_constraint"] = any(c not in REQUIRED_CONSTRAINTS
                                      for c in st.constraints)

        # --- plan wellformedness: a step whose required fact was never produced
        bad_pre = False
        for i, s in enumerate(st.plan[: st.cursor + 1]):
            need = s.get("requires")
            if need and need not in st.facts:
                bad_pre = True
        inv["plan_precondition"] = bad_pre

        # --- goal drift: the plan no longer targets the goal's output
        drift = False
        for s in st.plan:
            if s["tool"] == "write_file":
                p = s["args"].get("path")
                if isinstance(p, str) and p not in st.goal:
                    drift = True
        inv["goal_drift"] = drift

        # --- environment desync: the agent's cached view contradicts the env
        desync = False
        for path, view in st.env_view.items():
            if isinstance(view, str) and view.startswith("cached:"):
                claimed = view.split(":", 1)[1]
                actual = "%d rows" % max(0, self.env.fs.get(path, "").count("\n") - 1)
                if claimed.strip() != actual:
                    desync = True
        inv["state_desync"] = desync

        # --- grounding: every asserted fact must be supported by a real step
        unsupported, contradiction = False, False
        for name in st.facts:
            sup = st.fact_support.get(name)
            if sup is None:
                unsupported = True
            elif sup < 0:
                contradiction = True
        inv["unsupported_fact"] = unsupported
        inv["contradiction"] = contradiction

        inv["protocol_violation"] = False   # single-agent demo: F5 not exercised
        inv["role_conflict"] = False
        return inv

    # ------------------------------------------------------------- signals
    def observe(self, st: AgentState, step: Step) -> Signals:
        self.history.append(step)
        inv = self._invariants(st, step)

        critical = {"tool_unknown", "schema_invalid", "referent_missing",
                    "unguarded_write", "plan_precondition", "goal_drift"}
        hits = [k for k, v in inv.items() if v]
        I = 0.0
        if hits:
            I = min(1.0, 0.55 * sum(1 for h in hits if h in critical)
                    + 0.30 * sum(1 for h in hits if h not in critical))

        V = 0.0
        if inv["unsupported_fact"]:
            V += 0.6
        if inv["contradiction"]:
            V += 0.6
        if inv["retrieval_offtarget"]:
            V += 0.3
        V = min(1.0, V)

        # progress: has the plan cursor advanced recently?
        if st.cursor > self._last_cursor:
            self._last_cursor = st.cursor
            self._stalled_for = 0
        else:
            self._stalled_for += 1
        G = min(1.0, self._stalled_for / float(WINDOW))

        # redundancy: repeated action signatures inside the window
        win = self.history[-WINDOW:]
        sigs = [s.signature() for s in win if s.tool != "noop"]
        L = 0.0
        if len(sigs) >= 2:
            L = (len(sigs) - len(set(sigs))) / float(len(sigs) - 1)

        alpha = _sigmoid(W_I * I + W_V * V + W_G * G + W_L * L - BIAS + 0.5)
        sig = Signals(t=step.t, invariants=inv, I=round(I, 3), V=round(V, 3),
                      G=round(G, 3), L=round(L, 3), alpha=round(alpha, 3))

        # checkpoint the last state at which nothing was violated
        if not hits and step.ok:
            self.checkpoint = (st.clone(), self.env.snapshot(), step.t)
        return sig

    def reset_window(self) -> None:
        """Drop the evidence window after a rollback.

        The rolled-back steps are no longer part of the executed trajectory, so they
        must not contribute to the post-condition check of the repair that replaced
        them - otherwise the Verifier rejects a correct repair on stale evidence.
        """
        self.history = []
        self._stalled_for = 0
        self._last_cursor = 0

    @staticmethod
    def alarm(sig: Signals) -> bool:
        return sig.alpha > TAU_DET
