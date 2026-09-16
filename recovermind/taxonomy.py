"""The RecoverMind failure taxonomy: 5 classes, 16 subclasses.

Each subclass declares
  * the monitor signals it is expected to fire  -> used as the diagnostic evidence model
  * a discriminating predicate                  -> used to generate a safe probe
  * a canonical recovery operator               -> the cold-start prior for the Planner

This file is the single source of truth. The Diagnoser scores hypotheses by comparing
observed signals against `signals` here; it never hard-codes an answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Subclass:
    code: str
    name: str
    description: str
    signals: frozenset          # monitor signals this failure is expected to raise
    canonical: str              # canonical recovery operator
    secondary: Optional[str]    # next-best operator
    probe: Optional[str]        # discriminating predicate, evaluated by a read-only probe
    severity: str


def _s(*names) -> frozenset:
    return frozenset(names)


SUBCLASSES: Dict[str, Subclass] = {s.code: s for s in [
    # ---------------------------------------------------------------- F1
    Subclass("F1.1", "Goal misgeneralisation",
             "Actions optimise a proxy objective; the plan drifts from the stated goal.",
             _s("goal_drift", "no_progress"), "S3", "S2", "goal_matches_plan", "critical"),
    Subclass("F1.2", "Plan inconsistency",
             "A plan step's precondition was never established.",
             _s("plan_precondition", "tool_error"), "S2", "S3", "plan_wellformed", "high"),
    Subclass("F1.3", "Reasoning loop",
             "Near-duplicate action pairs recur without changing state.",
             _s("redundancy", "no_progress"), "S2", "S8", "state_advanced", "high"),
    Subclass("F1.4", "Anchoring",
             "An early framing dominates despite disconfirming observations.",
             _s("stale_constraint", "no_progress", "tool_error"), "S1", "S3", "constraint_supported",
             "medium"),
    # ---------------------------------------------------------------- F2
    Subclass("F2.1", "Unsupported claim",
             "An asserted fact is not entailed by any observation in the trace.",
             _s("unsupported_fact", "semantic_inconsistency"), "S5", "S1",
             "fact_has_support", "critical"),
    Subclass("F2.2", "Fabricated entity",
             "Reference to a file, API or record that does not exist.",
             _s("referent_missing", "tool_error"), "S5", "S4", "referent_resolves",
             "critical"),
    Subclass("F2.3", "Observation contradiction",
             "A claim directly negates a prior observation.",
             _s("contradiction", "semantic_inconsistency"), "S1", "S5",
             "fact_consistent", "high"),
    # ---------------------------------------------------------------- F3
    Subclass("F3.1", "Tool hallucination",
             "Invocation of a tool that is absent from the registry.",
             _s("tool_unknown"), "S4", None, "tool_in_registry", "critical"),
    Subclass("F3.2", "Parameter misspecification",
             "Correct tool, schema-invalid or semantically wrong arguments.",
             _s("schema_invalid"), "S4", None, "args_validate", "high"),
    Subclass("F3.3", "Execution fault",
             "Exception, error status, timeout or empty result from a tool.",
             _s("tool_error"), "S4", "S7", "tool_returns_ok", "high"),
    Subclass("F3.4", "Irreversibility blindness",
             "A destructive or externally visible action issued without a guard.",
             _s("unguarded_write"), "S7", "S9", "no_unguarded_write", "critical"),
    # ---------------------------------------------------------------- F4
    Subclass("F4.1", "Context overflow",
             "A task-critical constraint was evicted from the active window.",
             _s("constraint_missing", "token_limit"), "S6", "S1", "constraints_present",
             "high"),
    Subclass("F4.2", "Retrieval relevance failure",
             "Retrieved items are off-target or near-duplicate distractors.",
             _s("retrieval_offtarget"), "S6", "S5", "retrieval_relevant", "high"),
    Subclass("F4.3", "Temporal or state desync",
             "The agent's model of the world lags the actual environment state.",
             _s("state_desync", "contradiction"), "S6", "S7", "view_matches_env", "medium"),
    # ---------------------------------------------------------------- F5
    Subclass("F5.1", "Role ambiguity or conflict",
             "Two sub-agents claim the same subtask, or none does.",
             _s("role_conflict"), "S8", "S3", "unique_owner", "medium"),
    Subclass("F5.2", "Message or protocol fault",
             "Malformed, dropped or schema-violating message; deadlock.",
             _s("protocol_violation", "no_progress"), "S8", "S7", "protocol_ok", "high"),
]}

CLASS_NAMES = {
    "F1": "Planning and reasoning",
    "F2": "Grounding and hallucination",
    "F3": "Tool and action execution",
    "F4": "Memory and context",
    "F5": "Communication and coordination",
}

# Subclasses exercised by the single-agent demo environment. F5 needs the
# multi-agent scenario and is declared but not injectable here - see docs.
DEMO_SUBCLASSES: List[str] = [c for c in SUBCLASSES if not c.startswith("F5")]


def family(code: str) -> str:
    return code.split(".")[0]


def siblings(code: str) -> List[str]:
    f = family(code)
    return [c for c in SUBCLASSES if family(c) == f and c != code]


def canonical_operator(code: str) -> str:
    return SUBCLASSES[code].canonical
