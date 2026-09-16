# RecoverMind — Recovery Simulation

A visual, step-by-step simulation of RecoverMind's detect → diagnose → recover → verify
pipeline, built for presenting the system. This is a **separate demo layer** — it
does not modify `recovermind/`, `env/`, or the existing `webapp/` demo. It calls the real,
unmodified `RecoverMind` controller and renders its actual output; nothing is scripted or
faked.

## Run it

```bash
cd RecoverMind-System
python simulation/server.py
```

Opens automatically at `http://localhost:8877`. No installation, no API key, no internet —
standard library only, same as the rest of this project.

## Using it

1. **Pick a scenario** from the seven cards at the top (six real fault types plus a healthy
   control run — all seven are among the 14 fault subclasses actually implemented in
   `env/faults.py`).
2. **Click "▶ Start Simulation"**. The backend runs two real episodes (bare agent,
   RecoverMind) and the pipeline animates through what actually happened, step by step.
3. **Pause / Resume** at any point; **adjust playback speed** with the slider; **Reset**
   to try another scenario.
4. **Architecture View** (top right) shows the paper's actual Fig. 1 diagram plus the
   normal-agent-vs-recovery-loop distinction.

## What is real vs. what the demo layer adds

| In the UI | Source |
|---|---|
| Every number, diagnosis, strategy choice, and verification result | A real call to `RecoverMind(enabled=False/True).run(injected=...)` on the released package |
| Confidence values, ranked hypotheses, operator utilities | Parsed from the controller's own log output (`EpisodeReport.log`), not recomputed or approximated |
| Event log timestamps | Generated live by this demo as it plays back the (already-computed) trace — they mark playback time, not original inference latency. This scenario uses the deterministic scripted host (the same one behind the paper's Section IV study), so there is no LLM call to time in the first place. |
| "Control-Layer Decision Time" | Real `time.perf_counter()` measurement around the actual `.run()` call |
| The seven scenario labels ("Tool Execution Failure," etc.) | Plain-language names for real fault codes — the exact mapping is in `server.py`'s `CURATED_SCENARIOS` |

Nothing in this demo is marked "simulation-only," because everything shown is backed by a
real, currently-running computation.

## Files

```
simulation/
  server.py           backend: stdlib http.server, imports recovermind/ and env/ directly
  static/
    index.html         page structure
    style.css           styling (light/dark toggle)
    app.js               playback logic and rendering
    architecture.png    the paper's real Fig. 1, reused as-is
  README.md            this file
```
