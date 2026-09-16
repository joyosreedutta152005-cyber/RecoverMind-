"""The RecoverMind control loop (Algorithm 1).

    execute -> monitor -> [alarm] -> diagnose -> plan -> recover -> verify -> resume
                                        |                              |
                                     abstain                    budget exhausted
                                        |                              |
                                     continue                      escalate

Wraps an unmodified host agent. Set `enabled=False` to get the bare agent for the
side-by-side comparison, or use the ablation switches to disable diagnosis, memory
or verification independently.
"""
from __future__ import annotations

from typing import List, Optional

from env import agent as agent_mod
from env import faults, tools
from recovermind.diagnoser import FailureDiagnoser
from recovermind.executor import RecoveryExecutor
from recovermind.memory import RecoveryMemory
from recovermind.monitor import ExecutionMonitor
from recovermind.planner import RecoveryPlanner
from recovermind.strategies import LIBRARY
from recovermind.types import Attempt, Diagnosis, EpisodeReport, Step

MAX_STEPS = 45
BUDGET = 1.6           # total recovery cost allowed per task


class RecoverMind:
    def __init__(self, enabled: bool = True, memory: Optional[RecoveryMemory] = None,
                 use_diagnosis: bool = True, use_memory: bool = True,
                 verify: bool = True, r_max: float = 1.0, verbose: bool = False):
        self.enabled = enabled
        self.memory = memory if memory is not None else RecoveryMemory()
        self.use_diagnosis = use_diagnosis
        self.use_memory = use_memory
        self.verify = verify
        self.r_max = r_max
        self.verbose = verbose

    # ------------------------------------------------------------------ run
    def run(self, injected: Optional[str] = None, seed_env: Optional[tools.Environment] = None
            ) -> EpisodeReport:
        env = seed_env or tools.Environment()
        host = agent_mod.HostAgent(env)
        monitor = ExecutionMonitor(env)
        diagnoser = FailureDiagnoser(env, self.memory if self.use_memory else None)
        planner = RecoveryPlanner(self.memory if self.use_memory else None,
                                  r_max=self.r_max, use_memory=self.use_memory)
        executor = RecoveryExecutor(env, monitor, verify=self.verify)

        rep = EpisodeReport(task=agent_mod.TASK, injected=injected,
                            injected_at=faults.INJECT_AT.get(injected) if injected else None,
                            detected_at=None, diagnosed=None, diagnosis_correct=None)
        budget = BUDGET
        log = rep.log

        def say(msg: str) -> None:
            log.append(msg)
            if self.verbose:
                print(msg)

        for t in range(MAX_STEPS):
            if injected and rep.injected_at == host.state.cursor and not getattr(
                    self, "_done_inject", False):
                faults.inject(injected, host.state, env)
                self._done_inject = True
                say("  !! fault injected: %s at plan position %d" % (injected, host.state.cursor))

            if host.finished():
                break

            step = host.step()
            rep.trace.append(step)
            rep.steps_used += 1
            sig = monitor.observe(host.state, step)

            status = "ok " if step.ok else "ERR"
            say("  t=%-2d %s %-14s alpha=%.2f %s"
                % (step.t, status, step.tool, sig.alpha,
                   ("[" + ",".join(sig.fired()) + "]") if sig.fired() else ""))

            if not (self.enabled and monitor.alarm(sig)):
                continue

            # ---------------------------------------------------------- ALARM
            if rep.detected_at is None:
                rep.detected_at = host.state.cursor
            say("  -- ALARM  alpha=%.2f > tau_det" % sig.alpha)

            if not self.use_diagnosis:
                d = Diagnosis(subclass="F1.4", locus="n/a", confidence=0.0,
                              evidence=sig.fired())     # ablation: undifferentiated repair
                say("  -- diagnosis DISABLED, applying generic corrective step")
            else:
                d = diagnoser.diagnose(host.state, step, sig)
                if d is None:
                    say("  -- abstained (margin below threshold), continuing")
                    continue
                if rep.diagnosed is None:
                    rep.diagnosed = d.subclass
                    rep.diagnosis_correct = (injected is not None and d.subclass == injected)
                say("  -- DIAGNOSIS %s  conf=%.2f  probes=%d  evidence=%s"
                    % (d.subclass, d.confidence, d.probes_used, ",".join(d.evidence)))
                for c, p in d.ranked[:3]:
                    say("       cand %-5s p=%.2f" % (c, p))

            tried: List[str] = []
            recovered = False
            while budget > 0 and not recovered:
                ranked = planner.rank(d, budget, tried)
                for line in planner.explain(d, ranked):
                    say(line)
                sid = ranked[0][0]
                if sid == "S9":
                    break
                att, host.state = executor.attempt(sid, host.state, step, host)
                rep.attempts.append(att)
                rep.recovery_cost += att.cost
                budget -= att.cost
                tried.append(sid)
                say("  -- APPLY %s (%s) -> %s | %s"
                    % (sid, LIBRARY[sid].name, "VERIFIED" if att.verified else "rejected",
                       att.detail))
                if self.use_memory:
                    self.memory.record("|".join(sig.fired()), d.subclass, sid,
                                       1 if att.verified else 0,
                                       context="demo", step_cost=att.cost)
                recovered = att.verified

            if recovered:
                rep.recovered = True
                say("  == RECOVERED, resuming from checkpoint")
            else:
                rep.escalated = True
                say("  == BUDGET EXHAUSTED -> S9 escalate to human")
                break

        self._done_inject = False
        rep.task_success = host.task_succeeded()
        return rep
