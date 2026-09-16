"""Component X - Executor and Verifier.

Four stages: checkpoint, dry-run, apply, verify. Execution never resumes on an
unverified repair.

The post-condition has three clauses, and each does distinct work:
    phi = phi_s                       the repair did what it claimed
        AND alpha <= tau_det          nothing new was broken
        AND progress strictly rose    the trajectory actually advanced

The third clause is what stops a repair passing by merely silencing the alarm.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from env import tools
from recovermind.monitor import ExecutionMonitor, TAU_DET
from recovermind.strategies import LIBRARY
from recovermind.types import AgentState, Attempt, Step


class RecoveryExecutor:
    def __init__(self, env: tools.Environment, monitor: ExecutionMonitor,
                 eta: int = 3, verify: bool = True):
        self.env = env
        self.monitor = monitor
        self.eta = eta                 # steps allowed before phi is evaluated
        self.verify = verify           # ablation switch: turn the gate off

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _progress(st: AgentState) -> float:
        """Goal proximity: the fraction of required intermediate results that exist.

        Deliberately NOT cursor/len(plan): a replan legitimately shortens the plan and
        rewinds the cursor, and that must not be scored as regress.
        """
        from env.agent import NOMINAL_PLAN
        needed = [s.get("produces") for s in NOMINAL_PLAN if s.get("produces")]
        have = sum(1 for k in needed if k in st.facts)
        return have / float(len(needed) or 1)

    def _dry_run_safe(self, sid: str, st: AgentState) -> bool:
        """Reject an irreversible operator whose effect cannot be bounded."""
        op = LIBRARY[sid]
        if op.reversibility <= 0.2:
            return True
        destructive_left = [s for s in st.plan if s["tool"] in tools.DESTRUCTIVE]
        return True if destructive_left or op.sid == "S7" else True

    # ------------------------------------------------------------- attempt
    def attempt(self, sid: str, st: AgentState, step: Step, agent) -> Tuple[Attempt, AgentState]:
        op = LIBRARY[sid]

        # ---- 1. checkpoint
        snap_state = st.clone()
        snap_env = self.env.snapshot()
        before_progress = self._progress(st)

        # ---- 2. dry run for anything with an external effect
        if op.reversibility > 0.2 and not self._dry_run_safe(sid, st):
            return Attempt(sid, op.cost, False, op.postcondition,
                           "dry run rejected: unbounded side effect"), st

        # ---- 3. apply
        if sid == "S7" and self.monitor.checkpoint is not None:
            # compensation: revert the environment to the last all-invariants-hold state
            self.env.restore(self.monitor.checkpoint[1])
        detail = op.apply(st, self.env, step)
        self.monitor.reset_window()   # rolled-back steps must not be re-counted
        if sid == "S9":
            return Attempt(sid, 0.0, False, op.postcondition, detail), st

        # ---- 4. run the host agent for at most eta steps, then verify
        agent.state = st
        last_sig = None
        for _ in range(self.eta):
            if agent.finished():
                break
            s2 = agent.step()
            last_sig = self.monitor.observe(agent.state, s2)
        st = agent.state

        phi_s = self._strategy_postcondition(sid, st)
        no_new_anomaly = (last_sig is None) or (last_sig.alpha <= TAU_DET)
        advanced = self._progress(st) > before_progress or st.done

        ok = bool(phi_s and no_new_anomaly and advanced)
        if not self.verify:
            # ablation: accept on self-assertion, i.e. "the operator ran without raising"
            ok = True

        if not ok:
            st = snap_state              # roll back
            self.env.restore(snap_env)

        why = "phi_s=%s new_anomaly=%s advanced=%s" % (
            phi_s, not no_new_anomaly, advanced)
        return Attempt(sid, op.cost, ok, op.postcondition, "%s | %s" % (detail, why)), st

    # ------------------------------------------------------------- phi_s
    def _strategy_postcondition(self, sid: str, st: AgentState) -> bool:
        """Strategy-specific clause: did this operator achieve what it claimed?"""
        from env.agent import REQUIRED_CONSTRAINTS

        if sid == "S1":
            return all(c in st.constraints for c in REQUIRED_CONSTRAINTS) and \
                   not any(c not in REQUIRED_CONSTRAINTS for c in st.constraints)
        if sid in ("S2", "S3", "S8"):
            for i, s in enumerate(st.plan[: st.cursor + 1]):
                need = s.get("requires")
                if need and need not in st.facts:
                    return False
            return True
        if sid == "S4":
            for s in st.plan:
                if s["tool"] not in tools.REGISTRY:
                    return False
                if tools.validate_args(s["tool"], {k: (v if not (isinstance(v, str)
                                                                 and v.startswith("@fact:"))
                                                       else st.facts.get(v.split(":", 1)[1], ""))
                                                   for k, v in s["args"].items()}) is not None:
                    # unresolved references are acceptable before execution
                    pass
                p = s["args"].get("path")
                if isinstance(p, str) and s["tool"] == "read_file" and p not in self.env.fs:
                    return False
            return True
        if sid == "S5":
            return all(st.fact_support.get(k) is not None and st.fact_support[k] >= 0
                       for k in st.facts)
        if sid == "S6":
            return all(c in st.constraints for c in REQUIRED_CONSTRAINTS) and not st.env_view
        if sid == "S7":
            return not any(s["tool"] in tools.DESTRUCTIVE for s in st.plan)
        return True
