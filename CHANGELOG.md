# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-06-22

Initial public release.

### Changed
- **Renamed the project `CAG` → `RRL` (Retrieval Reputation Layer)** to avoid the naming
  collision with *Cache-Augmented Generation*. The importable package is now `rrl`
  (`from rrl import Retriever, CandidateStore, OutcomeSignals, ...`); the PyPI distribution
  name is `retrieval-reputation-layer`. Environment variables are now `RRL_DB_PATH`,
  `RRL_DECAY_UNIT_SEC`, `RRL_GAMMA`, and the default DB file is `rrl.db`.

### Added
- `pyproject.toml` (PEP 621) — installable via `pip install -e ".[api,llm,dev]"`.
- GitHub Actions CI running the test suite on Python 3.10–3.13.
- This `CHANGELOG.md` and `CONTRIBUTING.md`.
- The realistic recurrence harness (`sim/run_gate_recurring.py`) and its MBPP dataset
  (`data/mbpp.jsonl`) are now tracked in git, so README references resolve from a fresh clone.

### Fixed
- Portable per-call timeout in the LLM judge: replaced the Unix/main-thread-only
  `signal.SIGALRM` with a `ThreadPoolExecutor`-based timeout that works off the main thread
  (e.g. FastAPI's threadpool) and on Windows.
- Removed a duplicated `save_setting`/`get_setting` pair in the SQLite store.
- Pinned `requirements.txt` to the exact versions the benchmark gates were validated against.

[0.1.0]: https://github.com/pras-ops/retrieval-reputation-layer/releases/tag/v0.1.0
