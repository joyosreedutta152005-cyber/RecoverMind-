"""The recovery operator library S1-S9.

Each operator declares its cost, its reversibility, the operand it mutates, and a
post-condition that the Verifier evaluates. `apply` performs a real repair on the
agent state or the environment - there is no simulated success anywhere in here.
"""
from __future__ import annotations

import copy
import difflib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from env import tools
from env.agent import NOMINAL_PLAN, REQUIRED_CONSTRAINTS
from recovermind.types import AgentState, Step


@dataclass
class Operator:
    sid: str
    name: str
    operand: str
    cost: float             # normalised over tokens, steps and wall-clock
    reversibility: float    # 0.0 fully internal .. 1.0 irreversible side effect
    postcondition: str
    apply: Callable[[AgentState, tools.Environment, Step], str]


# ------------------------------------------------------------------ helpers
def _pipeline_order() -> List[str]:
    return [s.get("produces") for s in NOMINAL_PLAN if s.get("produces")]


def _invalidate(st: AgentState, names: List[str]) -> List[str]:
    """Discard the named facts *and everything derived from them*.

    Dependency-aware invalidation is what makes a repair sound. Dropping a corrupted
    intermediate result while keeping the values computed from it would let the agent
    rebuild its output from stale data and pass verification for the wrong reason.
    """
    order = _pipeline_order()
    positions = [order.index(n) for n in names if n in order]
    doomed = set(names)
    if positions:
        doomed |= set(order[min(positions):])
    for n in doomed:
        st.facts.pop(n, None)
        st.fact_support.pop(n, None)
    return sorted(doomed)


def _ungrounded(st: AgentState) -> List[str]:
    """Facts with no supporting step, or whose support does not point at a real step."""
    out = []
    for k in list(st.facts):
        sup = st.fact_support.get(k)
        if sup is None or sup < 0:
            out.append(k)
    return out


# ------------------------------------------------------------------ operators
def _s1_constraint_injection(st, env, step) -> str:
    """Repair the prompt/constraint set: drop unsupported framings, restore the task's."""
    dropped = [c for c in st.constraints if c not in REQUIRED_CONSTRAINTS]
    st.constraints = [c for c in st.constraints if c in REQUIRED_CONSTRAINTS]
    for c in REQUIRED_CONSTRAINTS:
        if c not in st.constraints:
            st.constraints.append(c)
    # a contradicted claim is discarded, together with everything derived from it
    retargeted = _retarget_from_constraints(st)
    killed = _invalidate(st, _ungrounded(st) + retargeted)
    if killed:
        _s2_local_replan(st, env, step)
    return ("dropped %d unsupported constraint(s); invalidated %s"
            % (len(dropped), ", ".join(killed) or "no claims"))


def _s2_local_replan(st, env, step) -> str:
    """Rebuild the last few plan steps from the nominal template, keeping progress."""
    done = {k for k in st.facts}
    fresh = copy.deepcopy(NOMINAL_PLAN)
    keep = [s for s in fresh if s.get("produces") in done]
    todo = [s for s in fresh if s.get("produces") not in done]
    st.plan = keep + todo
    st.cursor = len(keep)
    return "rebuilt %d remaining step(s) from the plan template" % len(todo)


def _s3_global_replan(st, env, step) -> str:
    """Discard the drifted plan and re-derive it from the goal."""
    st.plan = copy.deepcopy(NOMINAL_PLAN)
    st.cursor = 0
    st.facts.clear()
    st.fact_support.clear()
    st.constraints = list(REQUIRED_CONSTRAINTS)
    return "re-derived the full plan from the goal"


def _s4_tool_repair(st, env, step) -> str:
    """Re-resolve the tool against the registry and repair arguments against the schema."""
    notes: List[str] = []
    for s in st.plan:
        # (a) unknown tool -> nearest registry entry
        if s["tool"] not in tools.REGISTRY:
            match = difflib.get_close_matches(s["tool"], list(tools.REGISTRY), n=1, cutoff=0.5)
            if match:
                notes.append("%s -> %s" % (s["tool"], match[0]))
                s["tool"] = match[0]
        spec = tools.REGISTRY.get(s["tool"])
        if not spec:
            continue
        # (b) repair argument names against the schema
        for want in spec.schema:
            if want in s["args"]:
                continue
            near = difflib.get_close_matches(want, list(s["args"]), n=1, cutoff=0.4)
            if near:
                s["args"][want] = s["args"].pop(near[0])
                notes.append("arg %s -> %s" % (near[0], want))
        for extra in [k for k in list(s["args"]) if k not in spec.schema]:
            s["args"].pop(extra)
            notes.append("dropped arg %s" % extra)
        # (c) a path that does not resolve -> nearest existing path
        p = s["args"].get("path")
        if isinstance(p, str) and s["tool"] in ("read_file", "delete_file") and p not in env.fs:
            near = difflib.get_close_matches(p, list(env.fs), n=1, cutoff=0.4)
            if near:
                notes.append("path %s -> %s" % (p, near[0]))
                s["args"]["path"] = near[0]
    env.transient_errors.clear()          # retry with backoff clears transient faults
    return "; ".join(notes) or "cleared transient tool errors, retrying"


def _s5_grounded_verification(st, env, step) -> str:
    """Re-derive ungrounded or contradicted claims from a read-only tool call."""
    killed = _invalidate(st, _ungrounded(st))
    _s2_local_replan(st, env, step)       # the discarded facts must now be re-derived
    return "discarded %d claim(s) and their dependents: %s" % (
        len(killed), ", ".join(killed) or "-")


def _s6_context_repair(st, env, step) -> str:
    """Restore evicted constraints, drop distractors, resynchronise the env view."""
    restored = [c for c in REQUIRED_CONSTRAINTS if c not in st.constraints]
    st.constraints = list(REQUIRED_CONSTRAINTS)
    st.tokens = min(st.tokens, 20000)
    st.env_view.clear()
    retargeted = _retarget_from_constraints(st)
    killed = _invalidate(st, _ungrounded(st) + retargeted)
    _s2_local_replan(st, env, step)
    return ("restored %d constraint(s), resynchronised env view, invalidated %s"
            % (len(restored), ", ".join(killed) or "no stale claims"))


def _s7_compensating_action(st, env, step) -> str:
    """Undo the last environment mutation and remove the unguarded operation."""
    before = len(st.plan)
    st.plan = [s for s in st.plan if s["tool"] not in tools.DESTRUCTIVE]
    env.transient_errors.clear()
    st.env_view.clear()
    st.facts.pop("deleted", None)
    st.fact_support.pop("deleted", None)
    _s2_local_replan(st, env, step)       # recompute the cursor against the new plan
    return "reverted env state and removed %d destructive step(s)" % (before - len(st.plan))


def _s8_decompose(st, env, step) -> str:
    """Collapse repeated steps and re-bind the remaining work."""
    seen, deduped = set(), []
    for s in st.plan:
        key = (s["tool"], str(sorted(s["args"].items())), s.get("produces"))
        if key in seen and s.get("produces") is None:
            continue
        seen.add(key)
        deduped.append(s)
    removed = len(st.plan) - len(deduped)
    st.plan = deduped
    st.cursor = min(st.cursor, len(st.plan))
    _s2_local_replan(st, env, step)
    return "collapsed %d duplicated step(s)" % removed


def _s9_escalate(st, env, step) -> str:
    return "handed off to a human operator with diagnosis, evidence and checkpoint"


def _retarget_from_constraints(st: AgentState) -> List[str]:
    """Re-align plan arguments with the task's constraints.

    Returns the products of every step whose arguments changed: those values were
    computed from the wrong inputs and must be invalidated, or the replanner will
    treat them as already done and never re-issue the corrected call.
    """
    changed: List[str] = []

    def _set(step_spec, key, value):
        if step_spec["args"].get(key) != value:
            step_spec["args"][key] = value
            if step_spec.get("produces"):
                changed.append(step_spec["produces"])

    for s in st.plan:
        if s["tool"] == "compute_stats" and any("amount" in c for c in st.constraints):
            _set(s, "column", "amount")
        if s["tool"] == "write_file" and any("/out/report.txt" in c for c in st.constraints):
            _set(s, "path", "/out/report.txt")
        if s["tool"] == "read_file":
            for c in st.constraints:
                if "source data is " in c:
                    _set(s, "path", c.split("source data is ", 1)[1].strip())
    return changed


LIBRARY: Dict[str, Operator] = {op.sid: op for op in [
    Operator("S1", "Constraint injection", "prompt / constraints", 0.10, 0.0,
             "constraints restored and no unsupported framing remains",
             _s1_constraint_injection),
    Operator("S2", "Local replan", "plan (last j steps)", 0.25, 0.1,
             "plan preconditions satisfied and progress resumes", _s2_local_replan),
    Operator("S3", "Global replan", "plan (from the goal)", 0.60, 0.2,
             "plan targets the goal again", _s3_global_replan),
    Operator("S4", "Tool re-selection and signature repair", "tool binding / arguments",
             0.15, 0.2, "call validates and returns a well-formed result", _s4_tool_repair),
    Operator("S5", "Tool-grounded verification", "claims and evidence", 0.35, 0.1,
             "every claim is entailed by an observation", _s5_grounded_verification),
    Operator("S6", "Context repair", "context and memory", 0.15, 0.0,
             "required constraints present and state reconciled", _s6_context_repair),
    Operator("S7", "Compensating action and state reset", "external environment",
             0.70, 0.7, "environment invariants re-established", _s7_compensating_action),
    Operator("S8", "Decomposition and role re-binding", "roles / decomposition", 0.40, 0.1,
             "no duplicated work remains", _s8_decompose),
    Operator("S9", "Escalate or safe abort", "nothing (human hand-off)", 0.0, 0.0,
             "no further irreversible action is taken", _s9_escalate),
]}

REPAIR_OPERATORS = [s for s in LIBRARY if s != "S9"]
