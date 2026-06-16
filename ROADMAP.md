# CAG Feedback Layer (Retrieval Reputation Layer) Roadmap

This document outlines the roadmap for transitioning from core mathematical validation in simulation to a production-ready API and deployment model.

---

## Phase 1 — Core math, validated

**Exit criterion:** `c0` (Hidden Gem) converges to `C(i) > 0.8` within the run, and the unit tests pass. The math is now trustworthy.

| Step | Action | File |
|---|---|---|
| 1.1 | Fix the **Hidden-Gem starvation** (exploration over full set / rarity bonus / wider pre-cut pool) | `cag/retriever.py` |
| 1.2 | Implement the **ground-truth anchor** (override, not blend, when `s_gt` present) | `cag/feedback.py` |
| 1.3 | Implement **behavioral asymmetry** (keep capped ~0.75, regen sharp at 0.1) | `cag/feedback.py`, `sim/harness.py` |
| 1.4 | Add **unit tests** for the counter math: known signal sequence → asserted α/β/A/B | `tests/test_feedback.py` |

---

## Phase 2 — Prove it survives noise

**Exit criterion:** you can state, with a plot, "the design holds retrieval quality up to X% feedback noise and resists sycophantic skew."

| Step | Action | File |
|---|---|---|
| 2.1 | Add **sycophancy stress test** to the harness (biased "accept" decoupled from utility) | `sim/harness.py` |
| 2.2 | Add **noise sweep** (feedback noise 0→40%) — find where the Beta filter breaks | `sim/harness.py` |
| 2.3 | Plot **counter drift** with vs. without the asymmetry + GT-anchor | `sim/harness.py` |
| 2.4 | Write up the 2–3 numbers that come out (noise tolerance threshold, sycophancy resistance) | `walkthrough.md` |

---

## Phase 3 — Real retrieval (synthetic → actual documents)

**Exit criterion:** real queries against a real corpus return sensible top-k, and the feedback loop updates counters on real outcomes.

| Step | Action | File |
|---|---|---|
| 3.1 | Swap simulated sims for a real **embedding model** (e.g. a sentence-transformer or an API embedder) + real **BM25** | `cag/retriever.py` |
| 3.2 | Add a **chunking / ingestion** path — documents → candidates in the store | `cag/ingest.py` |
| 3.3 | Wire the **LLM judge** as `s_judge` for real (one cheap model, faithfulness prompt) | `cag/feedback.py` |
| 3.4 | Re-run retrieval quality on a small **real eval set** (20–50 labeled query/doc pairs) | `sim/harness.py` |

---

## Phase 4 — Persistence & service boundary

**Exit criterion:** restart the service, counters survive; two concurrent feedback calls don't corrupt α/β.

| Step | Action | File |
|---|---|---|
| 4.1 | Back the store with a **real DB** (vector store for embeddings + a table for α/β/A/B/timestamps) | `cag/store.py` |
| 4.2 | Make **decay lazy** (compute on read from `last_updated`) | `cag/store.py` |
| 4.3 | Wrap in an **API**: `POST /retrieve`, `POST /feedback` (FastAPI) | `cag/api.py` |
| 4.4 | Handle **concurrency** on counter updates (atomic increments / row locks) | `cag/store.py` |

---

## Phase 5 — Real feedback wiring

**Exit criterion:** real feedback flows in, shadow logs look sane, and the `y` distribution matches what you expected from simulation.

| Step | Action | File |
|---|---|---|
| 5.1 | Define the **`s_behave` capture contract** with the client | `cag/feedback.py` |
| 5.2 | Persist the **frozen `sim` credit shares `r(i)`** at answer time | `cag/store.py` |
| 5.3 | Add the **thumbs (`s_expl`)** and **verifier (`s_gt`)** hooks where they exist | `cag/feedback.py` |
| 5.4 | **Shadow mode**: log `y` and proposed counter updates *without applying them* | `cag/feedback.py` |

---

## Phase 6 — Production hardening & deploy

**Exit criterion:** live A/B shows the feedback loop improves retrieval quality without counter drift over a sustained window.

| Step | Action | File |
|---|---|---|
| 6.1 | **Observability**: log retrieval scores, `y`, `κ`, per-doc counter trajectories | `cag/api.py` |
| 6.2 | **Guardrails**: cap per-update counter movement; floor/ceiling on α,β to prevent runaway | `cag/feedback.py` |
| 6.3 | **Rollback / kill switch**: feature-flag the feedback updates | `cag/api.py` |
| 6.4 | **A/B harness**: feedback-on vs. feedback-off cohort to measure real lift | `cag/api.py` |
| 6.5 | Deploy (container + DB) | `Dockerfile`, `docker-compose.yml` |
