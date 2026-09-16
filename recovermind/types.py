"""Core data types shared by every RecoverMind component."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Step:
    """One reason-act-observe cycle of the host agent."""
    t: int
    thought: str
    tool: str
    args: Dict[str, Any]
    observation: Any = None
    ok: bool = True
    error: Optional[str] = None

    def signature(self) -> str:
        return "%s(%s)" % (self.tool, ",".join(sorted(map(str, self.args.values()))))


@dataclass
class AgentState:
    """Everything RecoverMind is allowed to observe or repair."""
    goal: str
    plan: List[Dict[str, Any]] = field(default_factory=list)
    cursor: int = 0
    constraints: List[str] = field(default_factory=list)   # working context
    facts: Dict[str, Any] = field(default_factory=dict)    # asserted claims
    fact_support: Dict[str, int] = field(default_factory=dict)  # fact -> step that grounds it
    env_view: Dict[str, Any] = field(default_factory=dict)      # agent's belief about env
    tokens: int = 0
    done: bool = False

    def clone(self) -> "AgentState":
        import copy
        return copy.deepcopy(self)


@dataclass
class Signals:
    """Monitor output for one step. Every field is computed, never guessed."""
    t: int
    invariants: Dict[str, bool] = field(default_factory=dict)  # name -> violated?
    I: float = 0.0      # invariant severity
    V: float = 0.0      # semantic inconsistency
    G: float = 0.0      # progress stagnation
    L: float = 0.0      # redundancy
    alpha: float = 0.0  # composite anomaly score

    def fired(self) -> List[str]:
        """Names of the signals that are actually active - the evidence for diagnosis."""
        out = [k for k, v in self.invariants.items() if v]
        if self.V > 0.5:
            out.append("semantic_inconsistency")
        if self.G > 0.5:
            out.append("no_progress")
        if self.L > 0.5:
            out.append("redundancy")
        return out


@dataclass
class Diagnosis:
    subclass: str                     # e.g. "F3.1"
    locus: str                        # e.g. "step:4/tool:read_flie"
    confidence: float                 # posterior margin
    ranked: List[Tuple[str, float]] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    probes_used: int = 0

    @property
    def failure_class(self) -> str:
        return self.subclass.split(".")[0]


@dataclass
class Attempt:
    strategy: str
    cost: float
    verified: bool
    postcondition: str
    detail: str = ""


@dataclass
class Case:
    """One entry of the Recovery Memory. Admitted only when verified."""
    signature: str
    subclass: str
    strategy: str
    outcome: int          # 1 verified success, 0 verified failure
    context: str
    step_cost: float
    inserted_at: int


@dataclass
class EpisodeReport:
    task: str
    injected: Optional[str]          # ground-truth subclass, if a fault was injected
    injected_at: Optional[int]
    detected_at: Optional[int]
    diagnosed: Optional[str]
    diagnosis_correct: Optional[bool]
    attempts: List[Attempt] = field(default_factory=list)
    recovered: bool = False
    escalated: bool = False
    task_success: bool = False
    steps_used: int = 0
    recovery_cost: float = 0.0
    trace: List[Step] = field(default_factory=list)
    log: List[str] = field(default_factory=list)

    @property
    def detection_lag(self) -> Optional[int]:
        if self.injected_at is None or self.detected_at is None:
            return None
        return self.detected_at - self.injected_at
