# Contributing to RRL

Thanks for your interest. RRL (Retrieval Reputation Layer) is a research/experimental
project built around **honest evaluation** — contributions that strengthen, falsify, or
extend the validated claims are especially welcome.

## Dev setup

```bash
git clone https://github.com/pras-ops/retrieval-reputation-layer
cd retrieval-reputation-layer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[api,llm,dev]"     # core + API + LLM judge + sims/tests
```

> The first real retrieval downloads `all-MiniLM-L6-v2` (~80 MB). The LLM judge needs
> `GEMINI_API_KEY` or Vertex AI credentials; without them it falls back to a local heuristic.

## Running tests and simulations

```bash
python -m pytest tests/ -q                 # unit tests (35) — network-free, model is mocked
python sim/verify_robustness.py            # robustness ablation (20-seed)
python sim/run_gate_a.py                   # value under recurrence (10-seed)
python sim/run_gate_b.py                   # decay / freshness (30-seed)
python sim/run_gate_recurring.py --selftest  # realistic benchmark: offline plumbing check
```

## Guidelines

- **Keep the evaluation honest.** New claims need a reproducible gate (multi-seed, with CIs)
  and must state their scope and failure mode, in the spirit of the existing `Validation Status`.
- **Don't move a result from "NOT yet validated" to "Validated" without the gate to back it.**
- Match the surrounding code style; add/extend unit tests for any behavior change.
- Run `python -m pytest tests/ -q` before opening a PR — CI runs the same on 3.10–3.13.

## Reporting issues

Open a GitHub issue with a minimal repro. For evaluation disputes, include the gate, seed
count, and the numbers you observed versus those claimed.
