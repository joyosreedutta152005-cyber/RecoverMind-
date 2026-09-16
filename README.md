# RecoverMind — Interactive Recovery Simulation

A visual, step-by-step demonstration of RecoverMind: a control layer that wraps an
unmodified LLM agent to detect a mid-execution failure, diagnose its typed root cause,
select and apply a repair strategy, and verify the repair before letting the agent
resume.

This repository is the **implementation + interactive demo only**. It does not include
the research paper or the broader experimental audit.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Run it

No installation, no API key, no internet — Python 3.8+ and the standard library only.

```bash
python faculty_demo/server.py
```

Opens at `http://localhost:8877`. Pick a failure scenario, click **Start Simulation**,
and watch the pipeline animate through detection → diagnosis → recovery → verification.

## What this actually is

`faculty_demo/` is a browser UI over the real `recovermind/` control layer and the real
`env/` task environment — it is not a mockup. Every number shown (anomaly scores,
diagnosis confidence, ranked alternative hypotheses, the chosen repair operator, the
three post-condition checks, before/after step counts) comes from an actual call to
`RecoverMind(enabled=False/True).run(injected=...)` on the code in this repository, run
live when you click Start. Nothing is scripted or pre-recorded.

Each of the 7 scenario cards is a real, implemented fault type from the taxonomy in
`recovermind/taxonomy.py` / `env/faults.py` — not an illustrative label. Selecting one
also shows the ground-truth plan position the fault is declared to corrupt
(`env/faults.py`'s own `INJECT_AT` table), and the live run confirms where it actually
landed, so you can see the two agree.

Full detail on what's real vs. how it's presented: `faculty_demo/README.md`.

## Structure

```
recovermind/     the control layer: Monitor, Diagnoser, Planner, Executor/Verifier, Memory
env/              the task environment: a real (in-memory) tool layer + 14 fault injectors
faculty_demo/    the browser UI: server.py (stdlib http.server) + static/ (HTML/CSS/JS)
requirements.txt  Python 3.8+, stdlib only (pytest is an optional dev-only extra)
```

## License

MIT — see `LICENSE`.
