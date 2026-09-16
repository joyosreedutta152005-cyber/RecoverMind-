"""Regression tests. Run with:  python -m pytest tests -q      (or)  python tests/test_system.py

These lock in the properties that actually matter, several of which were real bugs
found during development and are documented in docs/HOW_IT_WORKS.md.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import tools
from env.agent import HostAgent
from recovermind.controller import RecoverMind
from recovermind.memory import RecoveryMemory
from recovermind.strategies import LIBRARY, REPAIR_OPERATORS
from recovermind.taxonomy import DEMO_SUBCLASSES, SUBCLASSES


# ------------------------------------------------------------------ environment
def test_agent_completes_clean_task():
    assert RecoverMind(enabled=False).run(injected=None).task_success


def test_every_fault_actually_breaks_the_agent():
    """A fault that the bare agent survives is not a fault, and would flatter us."""
    survived = [c for c in DEMO_SUBCLASSES
                if RecoverMind(enabled=False).run(injected=c).task_success]
    # F3.3 is a transient error the plain agent can outlast; everything else must break it.
    assert survived == ["F3.3"], survived


def test_tools_never_leak_raw_exceptions():
    env = tools.Environment()
    for bad in [("nope", {}), ("read_file", {"path": "/missing"}),
                ("compute_stats", {"rows": [{"a": "x"}], "column": "a"})]:
        try:
            tools.call(env, bad[0], bad[1])
        except tools.ToolError:
            pass
        except Exception as e:                     # noqa: BLE001
            raise AssertionError("leaked %s: %s" % (type(e).__name__, e))


# ------------------------------------------------------------------ monitor
def test_no_false_alarm_on_healthy_run():
    rep = RecoverMind(enabled=True).run(injected=None)
    assert not rep.attempts, "monitor fired on a healthy trajectory"
    assert rep.task_success


def test_monitor_detects_every_fault():
    for c in DEMO_SUBCLASSES:
        rep = RecoverMind(enabled=True).run(injected=c)
        assert rep.detected_at is not None, "no alarm for %s" % c


# ------------------------------------------------------------------ diagnoser
def test_diagnosis_accuracy_cold_memory():
    hit = sum(bool(RecoverMind(enabled=True).run(injected=c).diagnosis_correct)
              for c in DEMO_SUBCLASSES)
    assert hit == len(DEMO_SUBCLASSES), "%d/%d" % (hit, len(DEMO_SUBCLASSES))


def test_memory_does_not_corrupt_diagnosis():
    """Regression: an over-confident prior once rewrote the diagnosis it was built from."""
    mem = RecoveryMemory()
    for _ in range(5):
        for c in DEMO_SUBCLASSES:
            RecoverMind(enabled=True, memory=mem).run(injected=c)
    hit = sum(bool(RecoverMind(enabled=True, memory=mem).run(injected=c).diagnosis_correct)
              for c in DEMO_SUBCLASSES)
    assert hit == len(DEMO_SUBCLASSES), "warm memory degraded diagnosis to %d" % hit


# ------------------------------------------------------------------ recovery
def test_recovers_every_fault():
    ok = sum(RecoverMind(enabled=True).run(injected=c).task_success
             for c in DEMO_SUBCLASSES)
    assert ok == len(DEMO_SUBCLASSES), "%d/%d" % (ok, len(DEMO_SUBCLASSES))


def test_recovery_stays_within_budget():
    for c in DEMO_SUBCLASSES:
        rep = RecoverMind(enabled=True).run(injected=c)
        assert rep.recovery_cost <= 1.6 + 1e-9, "%s overspent: %.2f" % (c, rep.recovery_cost)


# ------------------------------------------------------------------ memory
def test_memory_stores_only_verified_outcomes():
    mem = RecoveryMemory()
    RecoverMind(enabled=True, memory=mem).run(injected="F3.1")
    assert mem.cases
    assert all(c.outcome in (0, 1) for c in mem.cases)


def test_memory_round_trips(tmp="_test_mem.json"):
    mem = RecoveryMemory()
    for c in DEMO_SUBCLASSES[:4]:
        RecoverMind(enabled=True, memory=mem).run(injected=c)
    mem.save(tmp)
    again = RecoveryMemory(tmp)
    assert len(again.cases) == len(mem.cases)
    os.remove(tmp)


# ------------------------------------------------------------------ safety
def test_irreversibility_ceiling_is_respected():
    """With r_max below S7's reversibility, S7 must never be applied."""
    for c in DEMO_SUBCLASSES:
        rep = RecoverMind(enabled=True, r_max=0.2).run(injected=c)
        assert all(a.strategy != "S7" for a in rep.attempts), \
            "S7 used for %s despite the ceiling" % c


def test_library_covers_every_subclass():
    for code, sc in SUBCLASSES.items():
        assert sc.canonical in LIBRARY, "%s has no canonical operator" % code


def test_no_operator_is_redundant():
    """Every repair operator is the canonical choice for at least one subclass."""
    used = {sc.canonical for sc in SUBCLASSES.values()}
    unused = [s for s in REPAIR_OPERATORS if s not in used]
    assert not unused, "operators no subclass needs: %s" % unused


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print("PASS  %s" % name)
        except AssertionError as e:
            failed += 1
            print("FAIL  %s\n      %s" % (name, e))
    print("\n%d passed, %d failed, %d total" % (len(fns) - failed, failed, len(fns)))
    sys.exit(1 if failed else 0)
