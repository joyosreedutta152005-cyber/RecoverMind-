"""A small, real tool layer over an in-memory filesystem.

Everything here genuinely executes: schema validation, registry lookup and error
returns are real computations, which is what lets the Monitor's invariants and the
Verifier's post-conditions be honest rather than simulated.
"""
from __future__ import annotations

import io
import csv
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class ToolError(Exception):
    pass


@dataclass
class ToolSpec:
    name: str
    fn: Callable
    schema: Dict[str, type]      # arg name -> expected python type
    read_only: bool              # read-only tools are admissible as diagnostic probes
    description: str


class Environment:
    """In-memory filesystem plus a fault-injectable transient-error channel."""

    def __init__(self) -> None:
        self.fs: Dict[str, str] = {}
        self.writes: List[Dict[str, Any]] = []      # audit log, enables S7 compensation
        self.transient_errors: Dict[str, int] = {}  # tool -> remaining forced failures
        self.seed_data()

    def seed_data(self) -> None:
        self.fs["/data/sales.csv"] = (
            "region,amount,units\n"
            "north,1200,30\n"
            "south,900,22\n"
            "east,1500,41\n"
            "west,700,18\n"
        )
        self.fs["/data/README.txt"] = "Quarterly sales export. Amounts are in USD."

    def snapshot(self) -> Dict[str, Any]:
        return {"fs": dict(self.fs), "writes": list(self.writes)}

    def restore(self, snap: Dict[str, Any]) -> None:
        self.fs = dict(snap["fs"])
        self.writes = list(snap["writes"])


# --------------------------------------------------------------------------- tools
def _list_dir(env: Environment, path: str) -> List[str]:
    if not path.startswith("/"):
        raise ToolError("path must be absolute")
    hits = [p for p in env.fs if p.rsplit("/", 1)[0] == path.rstrip("/")]
    if not hits:
        raise ToolError("no such directory: %s" % path)
    return sorted(hits)


def _read_file(env: Environment, path: str) -> str:
    if path not in env.fs:
        raise ToolError("no such file: %s" % path)
    return env.fs[path]


def _write_file(env: Environment, path: str, content: str) -> bool:
    env.fs[path] = content
    env.writes.append({"path": path, "bytes": len(content)})
    return True


def _delete_file(env: Environment, path: str) -> bool:
    """Destructive and irreversible without compensation - the F3.4 hazard."""
    if path not in env.fs:
        raise ToolError("no such file: %s" % path)
    del env.fs[path]
    env.writes.append({"path": path, "deleted": True})
    return True


def _parse_csv(env: Environment, text: str) -> List[Dict[str, str]]:
    if not isinstance(text, str) or "," not in text:
        raise ToolError("input does not look like CSV")
    return list(csv.DictReader(io.StringIO(text)))


def _compute_stats(env: Environment, rows: List[Dict[str, str]], column: str) -> Dict[str, float]:
    if not isinstance(rows, list) or not rows:
        raise ToolError("rows must be a non-empty list")
    if column not in rows[0]:
        raise ToolError("column %r not present; have %s" % (column, list(rows[0])))
    try:
        vals = [float(r[column]) for r in rows]
    except (TypeError, ValueError):
        raise ToolError("column %r is not numeric" % column)
    return {"n": len(vals), "total": sum(vals), "mean": round(sum(vals) / len(vals), 2),
            "max": max(vals), "min": min(vals)}


def _format_report(env: Environment, stats: Dict[str, float], title: str) -> str:
    if not isinstance(stats, dict) or "total" not in stats:
        raise ToolError("stats must be a dict produced by compute_stats")
    return ("%s\n%s\nrecords : %d\ntotal   : %.2f\nmean    : %.2f\nmax     : %.2f\n"
            % (title, "=" * len(title), stats["n"], stats["total"],
               stats["mean"], stats["max"]))


def _lookup_schema(env: Environment, name: str) -> Dict[str, Any]:
    """Read-only introspection tool. Used by the Diagnoser as a probe."""
    spec = REGISTRY.get(name)
    if spec is None:
        return {"exists": False}
    return {"exists": True, "schema": {k: v.__name__ for k, v in spec.schema.items()},
            "read_only": spec.read_only}


def _file_exists(env: Environment, path: str) -> Dict[str, Any]:
    """Read-only existence probe - discriminates F2.2 from F3.3."""
    return {"exists": path in env.fs, "path": path}


REGISTRY: Dict[str, ToolSpec] = {}


def _register(name, fn, schema, read_only, description):
    REGISTRY[name] = ToolSpec(name, fn, schema, read_only, description)


_register("list_dir", _list_dir, {"path": str}, True, "List files under a directory")
_register("read_file", _read_file, {"path": str}, True, "Read a file's contents")
_register("write_file", _write_file, {"path": str, "content": str}, False, "Write a file")
_register("delete_file", _delete_file, {"path": str}, False, "Delete a file (destructive)")
_register("parse_csv", _parse_csv, {"text": str}, True, "Parse CSV text into rows")
_register("compute_stats", _compute_stats, {"rows": list, "column": str}, True,
          "Summary statistics for a numeric column")
_register("format_report", _format_report, {"stats": dict, "title": str}, True,
          "Render statistics as a report")
_register("lookup_schema", _lookup_schema, {"name": str}, True, "Introspect a tool's schema")
_register("file_exists", _file_exists, {"path": str}, True, "Check whether a path exists")

DESTRUCTIVE = {"delete_file"}


def validate_args(tool: str, args: Dict[str, Any]) -> Optional[str]:
    """Return None if the call is schema-valid, else a human-readable reason."""
    spec = REGISTRY.get(tool)
    if spec is None:
        return "tool %r is not in the registry" % tool
    missing = [k for k in spec.schema if k not in args]
    if missing:
        return "missing argument(s): %s" % ", ".join(missing)
    extra = [k for k in args if k not in spec.schema]
    if extra:
        return "unexpected argument(s): %s" % ", ".join(extra)
    for k, want in spec.schema.items():
        if not isinstance(args[k], want):
            return "argument %r should be %s, got %s" % (
                k, want.__name__, type(args[k]).__name__)
    return None


def call(env: Environment, tool: str, args: Dict[str, Any]):
    """Execute a tool. Raises ToolError on any failure - never returns a bad value."""
    spec = REGISTRY.get(tool)
    if spec is None:
        raise ToolError("unknown tool: %s" % tool)
    problem = validate_args(tool, args)
    if problem:
        raise ToolError(problem)
    if env.transient_errors.get(tool, 0) > 0:
        env.transient_errors[tool] -= 1
        raise ToolError("service unavailable (503) from %s" % tool)
    return spec.fn(env, **args)
