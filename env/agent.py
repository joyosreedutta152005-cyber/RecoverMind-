"""A deterministic ReAct-style host agent.

RecoverMind treats this as a black box observed through a read-only trace tap. The
agent is deliberately simple and honest: it follows a plan, calls real tools, records
what it believes, and has no self-correction of its own. Every recovery you see in
the demo comes from RecoverMind, not from the agent.
"""
from __future__ import annotations

import sys, os
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import tools
from recovermind.types import AgentState, Step

TASK = "Write a sales summary report to /out/report.txt from /data/sales.csv"

#: The nominal plan. Steps name the tool and how to source each argument:
#: a literal value, or `@fact:<name>` meaning "the fact produced earlier".
NOMINAL_PLAN: List[Dict[str, Any]] = [
    {"tool": "list_dir",      "args": {"path": "/data"},                         "produces": "listing"},
    {"tool": "read_file",     "args": {"path": "/data/sales.csv"},               "produces": "csv_text",
     "requires": "listing"},
    {"tool": "parse_csv",     "args": {"text": "@fact:csv_text"},                "produces": "rows",
     "requires": "csv_text"},
    {"tool": "compute_stats", "args": {"rows": "@fact:rows", "column": "amount"},"produces": "stats",
     "requires": "rows"},
    {"tool": "format_report", "args": {"stats": "@fact:stats", "title": "Sales Summary"},
     "produces": "report", "requires": "stats"},
    {"tool": "write_file",    "args": {"path": "/out/report.txt", "content": "@fact:report"},
     "produces": "written", "requires": "report"},
]

REQUIRED_CONSTRAINTS = [
    "report must be written to /out/report.txt",
    "summarise the 'amount' column",
    "source data is /data/sales.csv",
]


def fresh_state() -> AgentState:
    import copy
    return AgentState(
        goal=TASK,
        plan=copy.deepcopy(NOMINAL_PLAN),
        cursor=0,
        constraints=list(REQUIRED_CONSTRAINTS),
        facts={},
        fact_support={},
        env_view={},
    )


class HostAgent:
    """Executes one plan step per call. No internal recovery."""

    def __init__(self, env: tools.Environment, state: Optional[AgentState] = None):
        self.env = env
        self.state = state or fresh_state()
        self.t = 0

    # ------------------------------------------------------------------ helpers
    def _resolve(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith("@fact:"):
            key = value.split(":", 1)[1]
            return self.state.facts.get(key, "<<unresolved:%s>>" % key)
        return value

    def finished(self) -> bool:
        return self.state.done or self.state.cursor >= len(self.state.plan)

    # ------------------------------------------------------------------ one step
    def step(self) -> Step:
        st = self.state
        if self.finished():
            st.done = True
            return Step(self.t, "task complete", "noop", {}, observation=None, ok=True)

        spec = st.plan[st.cursor]
        args = {k: self._resolve(v) for k, v in spec["args"].items()}
        thought = "step %d/%d: %s" % (st.cursor + 1, len(st.plan), spec["tool"])
        step = Step(self.t, thought, spec["tool"], args)

        try:
            obs = tools.call(self.env, spec["tool"], args)
            step.observation = obs
            step.ok = True
            produced = spec.get("produces")
            if produced:
                st.facts[produced] = obs
                st.fact_support[produced] = self.t     # grounded by this step
            if spec["tool"] in ("write_file", "delete_file"):
                st.env_view[args.get("path", "?")] = "written"
            st.cursor += 1
        except tools.ToolError as e:
            step.ok = False
            step.error = str(e)
            # A plain ReAct agent does not know how to fix this; it stalls here.

        st.tokens += 40 + 12 * len(str(args))
        self.t += 1
        return step

    # ------------------------------------------------------------------ outcome
    def task_succeeded(self) -> bool:
        """Ground truth: recompute the correct answer from the environment and compare.

        Deliberately not a hard-coded number: some faults legitimately change the source
        data, and a correct recovery must then produce a different correct answer.
        """
        out = self.env.fs.get("/out/report.txt")
        if not out or "Sales Summary" not in out:
            return False
        src = self.env.fs.get("/data/sales.csv", "")
        rows = [r for r in src.strip().splitlines()[1:] if r.strip()]
        try:
            expected = sum(float(r.split(",")[1]) for r in rows)
        except (IndexError, ValueError):
            return False
        return ("total   : %.2f" % expected) in out
