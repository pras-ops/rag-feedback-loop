# CAG — A Feedback-Learning Retrieval Layer

CAG is a lightweight layer that sits on top of a retrieval system and, over repeated use,
**learns which retrieved items actually produce good outcomes** — boosting what helps and
decaying what goes stale. It is built around per-document Beta counters updated from
feedback signals (verifier, user behavior, LLM judge, thumbs), with explicit safeguards
against noisy and sycophantic feedback.

> **What it is:** a usefulness-and-freshness tracker for closed-loop retrieval systems.
> **What it is not:** a truth detector. It is sharpest when a **verifier** (e.g. tests, a DB
> check) supplies part of the feedback, and it is explicitly *not* designed to defend against
> adversarial or unverifiable feedback at scale.

This is a **research/experimental** project. The mechanism, persistence layer, and robustness
safeguards are tested, and the core claim — *that feedback learning beats a static retriever* —
is now **validated on controlled benchmarks with a verifier** (Gate A) along with the
decay/freshness value-prop (Gate B). What remains open: a demonstration on a **real execution
verifier** (HumanEval — substrate built, full demo in progress) and the no-verifier regime
(see [Validation Status](#validation-status)).

---

## Where it fits

Two properties decide whether CAG helps: **trustworthy feedback** (a verifier, or a
controlled/trusted source) and **repetition** (similar queries recur enough for counters to
converge).

| Use case | Fit | Why |
|---|---|---|
| **Coding agents** (retrieve fix patterns / snippets) | **Strongest** | Hard verifier (tests pass/fail) + controlled corpus → clean, objective feedback |
| **Enterprise RAG over trusted docs** | Strong | Controlled source; main enemy is *staleness*, handled by decay + behavioral signal |
| **Internal tools/agents over controlled data** | Strong | Same logic as above |
| **Open web / public user-generated content** | **Avoid** | Adversarial + unverifiable feedback → the >50% identifiability wall (see Limitations) |

---

## Architecture

```
              ┌──────────────┐   text query    ┌──────────────────────────┐
  documents → │  ingest.py   │ ───────────────► │       retriever.py       │
              │ chunk+embed  │                  │  hybrid RRF (vec + BM25) │
              └──────────────┘                  │  + Beta exploration      │
                                                │  + C_robust exploitation │
                                                └────────────┬─────────────┘
                                                             │ top-k + credit shares r(i)
                          feedback (y)                       ▼
   ┌──────────────┐   ┌──────────────────┐        ┌────────────────────────┐
   │   judge.py   │──►│   feedback.py    │ ─────►  │  store.py / store_     │
   │ (faithfulness│   │ outcome y, κ     │ counters│  sqlite.py (persistent,│
   │  + fallback) │   │ liar counter,    │ update  │  atomic, lazy decay,   │
   └──────────────┘   │ robust estimators│        │  pending bridge)       │
                      └──────────────────┘        └────────────────────────┘
```

| Module | Responsibility |
|---|---|
| `cag/store.py` | `Candidate` dataclass (α/β, A/B, `fooled`/`verified`, `recent_outcomes`) + in-memory `CandidateStore` |
| `cag/store_sqlite.py` | Persistent store: durable, **lazy decay**, **atomic increments**, `pending` (retrieve↔feedback bridge), schema migration |
| `cag/retriever.py` | Hybrid retrieval (SentenceTransformer + custom BM25, RRF-fused), Thompson-sampling exploration, rarity bonus, ε-greedy, robust exploitation estimate |
| `cag/feedback.py` | Outcome aggregation `y`, soft κ-weighted update, liar counter, robust estimators, optional ADT denoising |
| `cag/judge.py` | LLM faithfulness judge (Gemini) with a token-overlap fallback when offline |
| `cag/ingest.py` | Document chunking + embedding into candidates |
| `cag/api.py` | FastAPI service: `POST /retrieve`, `POST /feedback`, `GET /health` |

---

## Install

Requires Python 3.10+.

```bash
pip install sentence-transformers numpy scipy scikit-learn       # core retrieval
pip install fastapi uvicorn pydantic                              # API
pip install google-genai                                         # optional: live LLM judge (else heuristic fallback)
pip install matplotlib                                            # simulations/plots
```

> The first retrieval downloads the `all-MiniLM-L6-v2` model (~80 MB). The LLM judge needs
> `GEMINI_API_KEY` or Vertex AI credentials; without them it falls back to a local heuristic.

---

## Quickstart (library)

```python
from cag.store import CandidateStore
from cag.ingest import Ingester
from cag.retriever import Retriever
from cag.feedback import OutcomeSignals, update_counters

store = CandidateStore()
ingester = Ingester()
ingester.ingest_document(store, "doc1", "Long document text ...")

# weights = (w_sim, w_c, w_p, w_explore)
retriever = Retriever(store, weights=(0.20, 0.40, 0.10, 0.30))

results = retriever.retrieve("my question", top_k=3, explore=True)
retrieved_sims = {cand.id: sim for cand, score, sim in results}

# After observing how the answer landed, feed an outcome back:
signals = OutcomeSignals(s_behave=0.9, s_gt=1.0, s_judge=0.8, s_expl=1.0)
from cag.feedback import calculate_outcome
y = calculate_outcome(signals)                 # y in [0,1]
update_counters(store, retrieved_sims, y, signals=signals)
```

## Quickstart (API)

```bash
uvicorn cag.api:app --reload     # uses SqliteCandidateStore at $CAG_DB_PATH (default cag.db)
```

```bash
# 1) retrieve — returns a response_id and freezes credit shares server-side
curl -X POST localhost:8000/retrieve -H 'content-type: application/json' \
  -d '{"query":"how do I avoid db anomalies?","top_k":3}'

# 2) feedback — references that response_id; updates counters atomically
curl -X POST localhost:8000/feedback -H 'content-type: application/json' \
  -d '{"response_id":"<id-from-step-1>","s_behave":0.9,"s_gt":1.0}'
```

`/retrieve` persists the frozen credit shares to the `pending` table; `/feedback` pops them
and applies the update through the store's **atomic** `increment()` — safe under concurrent
requests.

---

## How it works

**Ranking.** Each candidate is scored:
```
score(i) = w_sim·sim(i) + w_c·C_robust(i) + w_p·P(i)         # exploitation
         + w_explore·sim(i)·(ThompsonSample(α,β) + rarity)   # exploration (when explore=True)
```
- `sim(i)` — hybrid vector+BM25 relevance, RRF-fused and normalized.
- `C_robust(i)` — recent usefulness (Beta mean by default; see robust estimators).
- `P(i) = A/(A+B)` — long-term usefulness.
- Exploration is **scaled by `sim`** so it never surfaces wholly irrelevant docs.

**Outcome.** Feedback signals are aggregated into `y ∈ [0,1]`. If a verifier `s_gt` is present
it overrides (it's the one signal that can't be faked); otherwise a weighted mean of
`s_behave (0.45)`, `s_gt (0.30)`, `s_judge (0.15)`, `s_expl (0.10)`, renormalized over present signals.

**Update.** Decisiveness `κ = 2·|y−0.5|`; credit share `r(i)` from similarity (smoothed);
`α += κ·r·y`, `β += κ·r·(1−y)` (permanent A/B at a 0.25 rate). An ambiguous outcome (`y≈0.5`)
barely moves the counters; a decisive one moves them fully.

**Decay.** `x ← 1 + (x−1)·γ^Δt` pulls stale counters back toward the prior, computed lazily
from `last_updated` (no cron sweep).

---

## Robustness & denoising

Naive learning from implicit feedback can degrade — a known result in the literature. CAG
includes safeguards, evaluated in a 20-seed ablation (`sim/verify_robustness.py`):

| Mechanism | Status | Notes |
|---|---|---|
| **Behavioral cap** (positive `s_behave` ≤ 0.75) | **Adopted** | Asymmetric: trusts rejections fully, caps sycophantic "accepts" |
| **Verifier anchor** (`gt_override`) | **Adopted** | The one sycophancy-proof signal dominates when present |
| **Liar counter** (`fooled`/`verified` → per-doc `trust_score`) | **Adopted (default)** | Detects "accepted-but-verifier-failed"; lowest collateral damage to good docs |
| Trimmed mean (drop top 30%) | **Rejected** | Strong on contaminated data but biased *down* on clean data — craters good docs |
| Median-of-Means | **Rejected** | Block-averaging pre-mixes uniform contamination → ≈ the plain mean |
| ADT loss-downweighting | Optional, off by default | Helps *random* noise; does **not** help sycophancy (the lie is low-loss) |

**Honest bound:** these *mitigate* sycophancy, they do not *solve* it. Effectiveness is capped
by verifier coverage, and above ~50% contamination no estimator on the feedback values alone
can recover truth (information-theoretic). Robust estimator modes are selectable via
`robust_estimator_mode` (`"beta"` default, `"median"`, `"trimmed"`, `"mom"`).

---

## Validation Status

Reported honestly — what the tests/sims actually establish, and what they don't.

### Validated ✅
- **Persistence layer** (`tests/test_store_sqlite.py`): durability across reconnect, lazy
  decay math, **atomic concurrent increments** (8 threads × 200 increments, zero lost
  updates), and the pending retrieve↔feedback bridge.
- **API atomic path**: `/feedback` routes through `store.increment()`, not a Python
  read-modify-write — verified by reading the code path.
- **Robustness ablation** (20-seed, mean±std): supports adopting the liar counter and
  rejecting trimmed-mean / MoM, as above.
- **Gate A — value vs. static** (`sim/run_gate_a.py`): 10-seed sweep at top_k=1 with an
  **independent answer-verifier** that inspects only the generated answer text, never the
  retrieved doc IDs — so the training signal no longer leaks the eval label. CAG's late-stage
  answer correctness reaches **0.997 vs 0.700** for the static baseline, with **non-overlapping
  95% CIs**; the independent Recall@1 metric (never used for training) rises 0.57 → 0.85.
  *Scope:* controlled corpus, clean synthetic keyword-verifier, recurring queries — i.e. the
  verifier-present regime.
- **Gate B — decay / freshness** (`sim/run_gate_b.py`): 10-seed staleness sweep where ground
  truth flips at step 50. Decay-ON re-adapts (Phase-2 correctness **0.91 vs 0.45** for
  decay-OFF), non-overlapping CIs. *Scope:* controlled scenario, mock generation.
- **Gate C — real-verifier showcase** (`sim/run_gate_c.py`, `sim/gate_c_verifier.py`): executes real HumanEval unit
  tests on generated code in a sandboxed subprocess. Ingests a custom coding hint corpus (good vs distractor) on 5 problems. Under a 10-seed sweep using the real Gemini 2.5 Flash API, the Static retriever gets a pass rate of **34.6%** (partially bypassing distractors via pre-trained weights), while CAG learns to prioritize the good hints, achieving a late-stage pass rate of **44.7%** (and overall of **44.4%**) with non-overlapping 95% CIs on the overall metric.

### NOT yet validated ⚠️ (the important part)
- **No-verifier case — UNPROVEN.** Even Gate B and C used hard verifiers. CAG's behavior on purely
  behavioral/judge feedback (no `s_gt`) is not directly validated and is bounded by the
  robustness limits above (sycophancy is mitigated, not solved).
- **Real-traffic behavior** (degeneracy / popularity-bias amplification): the standard defense
  (exploration) is implemented but **not yet monitored**.

---

## Limitations & scope

- **Not a truth detector.** It tracks usefulness and freshness, not correctness.
- **Verifier-bounded.** Robustness against bad feedback rises and falls with how often a
  verifier (`s_gt`) is available.
- **The >50% wall.** If a majority of feedback for an item is dishonest, no statistic on the
  feedback alone recovers truth — by information theory. Scope CAG to controlled/verifiable
  settings.
- **Exploration cost.** Exploration improves discovery but can evict correct results from a
  small top-k; tune `epsilon`/`explore` to your top-k.
- **Not novel research.** This is a clean implementation of established ideas (online
  learning-to-rank with bandit feedback, Beta-Bernoulli reliability, recsys denoising). The
  intended value is a tidy, honestly-evaluated, drop-in layer — not a new algorithm.

---

## Repository layout

```
cag/                  core library (store, retriever, feedback, judge, ingest, api, store_sqlite)
sim/                  simulations & gates: harness.py, verify_isolation.py, verify_robustness.py,
                      run_gate_a.py (value), run_gate_b.py (decay), run_gate_c.py (HumanEval sweep),
                      gate_c_verifier.py (real tests)
data/                 HumanEval.jsonl (164 problems) — real verifier substrate for Gate C (sourced from OpenAI's HumanEval benchmark, MIT License)
tests/                test_feedback.py, test_store_sqlite.py, test_robustness.py, test_api.py
ROADMAP.md            phased build plan
```

## Running tests & simulations

```bash
python3 -m unittest discover -s tests -p "test_*.py"   # unit tests
python3 sim/verify_robustness.py                       # 20-seed robustness ablation
python3 sim/run_gate_a.py                              # Gate A: value vs static (10-seed, unbiased verifier)
python3 sim/run_gate_b.py                              # Gate B: decay / freshness (10-seed staleness)
python3 sim/run_gate_c.py                              # Gate C: real HumanEval verifier showcase sweep
python3 sim/gate_c_verifier.py                         # Gate C verifier self-test
```

## Next steps

2. **Degeneracy monitoring** (retrieval concentration / coverage) before any real deployment.
3. **No-verifier validation** — CAG's behavior under purely behavioral/judge feedback.
4. Package as a pip-installable layer over an existing retriever interface.

See `ROADMAP.md` for the full plan.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

