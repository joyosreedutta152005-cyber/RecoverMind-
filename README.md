# RecoverMind

RecoverMind is a control layer that wraps an unmodified LLM agent to detect a
mid-execution failure, diagnose its typed root cause, select and apply a repair
strategy, and verify the repair before letting the agent resume.

This repository contains the **implementation, its measured results, and an
interactive visual demo**. It does not include the research paper text or the full
experimental audit — those are shared separately.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## See the results

**[`RESULTS.md`](RESULTS.md)** — measured outcomes across three studies (a 90-episode
live-LLM-host evaluation, a 14-fault scripted-host suite with a full ablation study, and
a 40-episode generalisation check), every number backed by a fresh, real re-execution
logged in `evidence/`.

## Run the interactive demo

No installation, no API key, no internet — Python 3.8+ and the standard library only.

```bash
python faculty_demo/server.py
```

Opens at `http://localhost:8877`. Pick a failure scenario, click **Start Simulation**,
and watch the pipeline animate through detection → diagnosis → recovery → verification.
Every number shown is computed live from an actual `RecoverMind.run()` call on the code
in this repository — nothing is scripted or pre-recorded. See `faculty_demo/README.md`
for detail.

## Reproduce the results yourself

```bash
python demo/run_benchmark.py     # the 14-fault suite + ablation study
python tests/test_system.py      # 14 regression tests
```

## Structure

```
recovermind/     the control layer: Monitor, Diagnoser, Planner, Executor/Verifier, Memory
env/              the task environment: a real (in-memory) tool layer + 14 fault injectors
faculty_demo/    the interactive browser demo
demo/             the benchmark + ablation study script behind RESULTS.md
tests/            14 regression tests
evidence/        fresh re-execution logs and result figures backing RESULTS.md
RESULTS.md       measured results, with reproduction instructions
requirements.txt  Python 3.8+, stdlib only (pytest is an optional dev-only extra)
```

## License

MIT — see `LICENSE`.
