# Contributing

Thanks for taking a look at fedcal. Keep changes small and measurable.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python -m fedcal.benchmark
```

## What belongs here

Good contributions improve the benchmark, make the algorithm easier to audit,
or add a small feature without adding a heavy dependency. Please include tests
for behavior changes and rerun the benchmark before updating any numbers in
the README.

## Style

Plain Python, typed where it helps, with comments only around non-obvious math
or benchmark assumptions. Do not add paid APIs, telemetry, hidden downloads, or
benchmark claims that cannot be reproduced locally.
