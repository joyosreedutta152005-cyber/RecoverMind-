# -*- coding: utf-8 -*-
"""RecoverMind recovery simulation - backend.

Separate from webapp/ on purpose (per the brief: keep the demo/visualisation layer apart
from both the core package and the existing webapp demo). Zero third-party dependencies,
same as the rest of RecoverMind-System: stdlib http.server around the *unmodified*
recovermind/ and env/ packages.

Every event log line parsed here comes verbatim from EpisodeReport.log, which the
controller already produces (controller.py's `say()` calls) for every real run. Nothing
about the control layer's decision rules is changed, re-implemented, or simulated - this
file runs two real episodes (bare agent, RecoverMind) per request and serialises their
real results to JSON. Wall-clock timing is measured with time.perf_counter() around the
actual RecoverMind.run() call, so it is real, not illustrative - but it measures the
scripted host's decision-making time, not LLM inference latency, since this demo uses the
same deterministic host as the paper's Section IV study, not the live-model study.

Run:
    python simulation/server.py
    (then open http://localhost:8877 )
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env import faults                                    # noqa: E402
from env.agent import NOMINAL_PLAN                          # noqa: E402
from recovermind.controller import RecoverMind, MAX_STEPS, BUDGET  # noqa: E402
from recovermind.strategies import LIBRARY                 # noqa: E402
from recovermind.taxonomy import SUBCLASSES, DEMO_SUBCLASSES, CLASS_NAMES, family  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
PORT = int(os.environ.get("PORT", "8877"))

# A curated subset of the 14 real, implemented fault subclasses, chosen to map cleanly
# onto plain-language scenario names. This is a subset for a readable dropdown, not a
# different or invented set of faults - all seven codes below are exactly as declared in
# recovermind/taxonomy.py and are fully injectable (see env/faults.py).
CURATED_SCENARIOS = [
    {"code": None,   "label": "No Failure (Healthy Control)",
     "note": "Runs the clean task with nothing injected, to show zero false alarms."},
    {"code": "F3.3", "label": "Tool Execution Failure",
     "note": "Covers exception / error status / timeout / empty result from a tool."},
    {"code": "F3.1", "label": "Tool Hallucination (Invalid Tool Call)",
     "note": "The agent calls a tool that is absent from the registry."},
    {"code": "F3.2", "label": "Parameter Misspecification (Invalid Input)",
     "note": "Correct tool, but schema-invalid or semantically wrong arguments."},
    {"code": "F1.3", "label": "Reasoning Loop (Repeated Failure)",
     "note": "Near-duplicate actions recur without the state changing."},
    {"code": "F2.2", "label": "Fabricated Entity (Invalid Reference)",
     "note": "Reference to a file, API, or record that does not exist."},
    {"code": "F3.4", "label": "Irreversibility Blindness",
     "note": "A destructive action issued without a guard - tests the reversibility-aware repair."},
]
CURATED_CODES = {s["code"] for s in CURATED_SCENARIOS if s["code"] is not None}
assert CURATED_CODES.issubset(set(DEMO_SUBCLASSES)), "curated scenario not in DEMO_SUBCLASSES"


def plan_tool_at(position):
    """The nominal plan's tool name at a given 0-indexed position, if in range."""
    if position is None or not (0 <= position < len(NOMINAL_PLAN)):
        return None
    return NOMINAL_PLAN[position]["tool"]

# ---------------------------------------------------------------- log parsing
# Same approach as webapp/server.py: parse the controller's existing human-readable
# narration rather than modify the released controller to also emit structured events.
RE_STEP = re.compile(r"t=\s*(\d+)\s+(ok|ERR)\s+(\S+)\s+alpha=([\d.]+)\s*(\[[^\]]*\])?")
RE_INJECT = re.compile(r"fault injected: (\S+) at plan position (\d+)")
RE_ALARM = re.compile(r"-- ALARM\s+alpha=([\d.]+)")
RE_DIAGNOSIS = re.compile(r"-- DIAGNOSIS (\S+)\s+conf=([\d.]+)\s+probes=(\d+)\s+evidence=(.*)")
RE_CAND = re.compile(r"cand\s+(\S+)\s+p=([\d.]+)")
RE_ABSTAIN = re.compile(r"-- abstained")
RE_RANK = re.compile(r"^(S\d)\s+U=([+-][\d.]+)\s+cost=([\d.]+)\s+rev=([\d.]+)\s+(.+?)\s+\((.+)\)$")
RE_APPLY = re.compile(r"-- APPLY (\S+) \((.+?)\) -> (VERIFIED|rejected) \| (.*)")
RE_VERIFY_DETAIL = re.compile(r"phi_s=(True|False)\s+new_anomaly=(True|False)\s+advanced=(True|False)")
RE_RECOVERED = re.compile(r"== RECOVERED")
RE_ESCALATE = re.compile(r"== BUDGET EXHAUSTED")


def parse_log(log):
    events = []
    for line in log:
        s = line.strip()
        m = RE_INJECT.search(s)
        if m:
            at = int(m.group(2))
            events.append({"kind": "inject", "fault": m.group(1), "at": at,
                            "tool": plan_tool_at(at)})
            continue
        m = RE_STEP.search(s)
        if m:
            events.append({"kind": "step", "t": int(m.group(1)), "ok": m.group(2) == "ok",
                            "tool": m.group(3), "alpha": float(m.group(4)),
                            "fired": (m.group(5) or "").strip("[]").split(",")
                                     if m.group(5) else []})
            continue
        m = RE_ALARM.search(s)
        if m:
            events.append({"kind": "alarm", "alpha": float(m.group(1))})
            continue
        m = RE_DIAGNOSIS.search(s)
        if m:
            events.append({"kind": "diagnosis", "subclass": m.group(1),
                            "confidence": float(m.group(2)), "probes": int(m.group(3)),
                            "evidence": [e for e in m.group(4).split(",") if e],
                            "candidates": []})
            continue
        m = RE_CAND.search(s)
        if m and events and events[-1]["kind"] == "diagnosis":
            events[-1]["candidates"].append({"subclass": m.group(1), "p": float(m.group(2))})
            continue
        if RE_ABSTAIN.search(s):
            events.append({"kind": "abstain"})
            continue
        m = RE_RANK.search(s)
        if m:
            events.append({"kind": "rank", "op": m.group(1), "utility": float(m.group(2)),
                            "cost": float(m.group(3)), "reversibility": float(m.group(4)),
                            "name": m.group(5).strip()})
            continue
        m = RE_APPLY.search(s)
        if m:
            v = {"kind": "apply", "op": m.group(1), "name": m.group(2),
                 "verified": m.group(3) == "VERIFIED", "detail": m.group(4)}
            vm = RE_VERIFY_DETAIL.search(m.group(4))
            if vm:
                v["verification"] = {
                    "phi_s": vm.group(1) == "True",
                    "no_new_anomaly": vm.group(2) == "False",  # note: field itself is "new_anomaly"
                    "advanced": vm.group(3) == "True",
                }
            events.append(v)
            continue
        if RE_RECOVERED.search(s):
            events.append({"kind": "recovered"})
            continue
        if RE_ESCALATE.search(s):
            events.append({"kind": "escalate"})
            continue
    return events


def step_to_dict(step):
    return {"t": step.t, "thought": step.thought, "tool": step.tool, "args": step.args,
            "ok": step.ok, "error": step.error}


def attempt_to_dict(a):
    return {"strategy": a.strategy, "name": LIBRARY[a.strategy].name if a.strategy in LIBRARY
            else a.strategy, "cost": a.cost, "verified": a.verified,
            "postcondition": a.postcondition, "detail": a.detail}


def report_to_dict(rep, elapsed_s):
    return {
        "task": rep.task,
        "injected": rep.injected,
        "injected_at": rep.injected_at,
        "detected_at": rep.detected_at,
        "detection_lag": rep.detection_lag,
        "diagnosed": rep.diagnosed,
        "diagnosis_correct": rep.diagnosis_correct,
        "recovered": rep.recovered,
        "escalated": rep.escalated,
        "task_success": rep.task_success,
        "steps_used": rep.steps_used,
        "max_steps": MAX_STEPS,
        "recovery_cost": round(rep.recovery_cost, 3),
        "recovery_budget": BUDGET,
        "attempts": [attempt_to_dict(a) for a in rep.attempts],
        "trace": [step_to_dict(s) for s in rep.trace],
        "events": parse_log(rep.log),
        "elapsed_seconds": round(elapsed_s, 6),
    }


def run_pair(injected):
    """Bare agent and RecoverMind, same fault, same task. Real timing per arm."""
    t0 = time.perf_counter()
    bare = RecoverMind(enabled=False).run(injected=injected)
    t1 = time.perf_counter()
    rm = RecoverMind(enabled=True).run(injected=injected)
    t2 = time.perf_counter()
    return {"bare": report_to_dict(bare, t1 - t0), "recovermind": report_to_dict(rm, t2 - t1)}


def scenario_catalogue():
    out = []
    for s in CURATED_SCENARIOS:
        if s["code"] is None:
            out.append({"code": None, "label": s["label"], "note": s["note"],
                        "family": None, "family_name": None, "canonical": None,
                        "canonical_name": None, "severity": None,
                        "expected_at": None, "expected_tool": None})
            continue
        sc = SUBCLASSES[s["code"]]
        expected_at = faults.INJECT_AT.get(s["code"])
        out.append({
            "code": s["code"], "label": s["label"], "note": s["note"],
            "description": sc.description,
            "family": family(s["code"]), "family_name": CLASS_NAMES[family(s["code"])],
            "canonical": sc.canonical, "canonical_name": LIBRARY[sc.canonical].name,
            "severity": sc.severity,
            # Ground truth, declared before any episode runs: env/faults.py's own
            # INJECT_AT table says exactly which plan step this fault corrupts.
            "expected_at": expected_at,
            "expected_tool": plan_tool_at(expected_at),
        })
    return out


def operator_catalogue():
    return [{"id": sid, "name": op.name, "operand": op.operand, "cost": op.cost,
             "reversibility": op.reversibility, "postcondition": op.postcondition}
            for sid, op in LIBRARY.items()]


# ---------------------------------------------------------------- HTTP layer
class Handler(BaseHTTPRequestHandler):
    server_version = "RecoverMindSimulation/1.0"

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, path):
        if path == "/":
            path = "/index.html"
        fp = os.path.normpath(os.path.join(STATIC_DIR, path.lstrip("/")))
        if not fp.startswith(STATIC_DIR) or not os.path.isfile(fp):
            self.send_response(404); self.end_headers(); return
        ctype = {
            ".html": "text/html; charset=utf-8", ".js": "application/javascript",
            ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml",
        }.get(os.path.splitext(fp)[1], "application/octet-stream")
        with open(fp, "rb") as fh:
            data = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/scenarios":
            return self._json(scenario_catalogue())
        if path == "/api/operators":
            return self._json(operator_catalogue())
        return self._static(path)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/simulate":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(body or b"{}")
            except json.JSONDecodeError:
                payload = {}
            injected = payload.get("fault") or None
            if injected is not None and injected not in DEMO_SUBCLASSES:
                return self._json({"error": "unknown fault code: %r" % injected}, 400)
            try:
                result = run_pair(injected)
            except Exception as e:  # noqa: BLE001 - surface it to the UI instead of a 500 wall
                return self._json({"error": "%s: %s" % (type(e).__name__, e)}, 500)
            return self._json(result)
        self.send_response(404); self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = "http://127.0.0.1:%d" % PORT
    print("RecoverMind simulation running at %s  (Ctrl+C to stop)" % url)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
