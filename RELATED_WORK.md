# Related Work & Positioning

CAG sits at the intersection of two active (2024–2026) research areas and one older
statistical lineage. This document states where it fits, how it differs, and — honestly —
where it has *not* yet been compared.

> **Note:** citations below are for orientation. Verify exact venues/IDs against the source
> before any formal submission; a few are cited by name where the identifier was not confirmed.

---

## 1. Agent memory & experience reuse (the closest neighbour)

A wave of recent work lets agents **reuse verified past experience** to improve later — the
same high-level goal as CAG.

- **Evo-Memory** (Google DeepMind + UIUC, [arXiv:2511.20857](https://arxiv.org/abs/2511.20857)) — a
  *streaming* benchmark for test-time learning with self-evolving memory, where each
  interaction encodes *whether the task succeeded and which strategies worked*. Their **ReMem**
  framework adds a think–act–memory-refine loop. This is essentially CAG's
  recurrence-with-a-verifier regime, standardized.
- **ExpeL** (Zhao et al., AAAI 2024) — collects trajectories, abstracts insights, retrieves
  successful past experiences at test time.
- **Agent Workflow Memory** (Wang et al., 2024) — induces reusable *workflows* from past
  trajectories and injects them into context.
- **Dynamic Cheatsheet** (2025) — adaptive memory of reusable strategies/insights.
- **Agentic Context Engineering** ([arXiv:2510.04618](https://arxiv.org/pdf/2510.04618)) — evolves
  the context itself for self-improvement.

**How CAG differs.** All of the above operate at the **prompt-construction level**: they
*extract and inject* workflows, insights, or strategy text. CAG operates one level lower, at
the **ranking level** — it keeps a lightweight **per-document Beta reputation** (with decay and
sycophancy safeguards) and changes *which existing documents surface*. It writes no new
memory artifacts and puts no LLM in the memory loop. This is a smaller, cheaper, more auditable
mechanism — and a narrower claim. The flip side: CAG does *not* synthesize new strategies the
way ExpRAG / AWM / Dynamic Cheatsheet do.

---

## 2. Online / feedback-driven RAG reranking & learning-to-rank

CAG's update loop is online learning-to-rank with bandit feedback, applied to RAG.

- **DynamicRAG** ([arXiv:2505.07233](https://arxiv.org/pdf/2505.07233)) — RL-optimizes a reranker
  from LLM-output feedback.
- **AutoRAG-HP** — frames RAG knobs as hierarchical bandits tuned from live feedback.
- **LTRR: Learning to Rank Retrievers for LLMs** ([arXiv:2506.13743](https://arxiv.org/pdf/2506.13743)).
- **Online-Optimized RAG for tool use / function calling** ([arXiv:2509.20415](https://arxiv.org/pdf/2509.20415)).
- **REARANK** — reranking as an RL reasoning agent; **RankArena** — eval platform with
  human/LLM feedback.

**How CAG differs.** These largely *train or prompt a (neural / LLM) reranker* from feedback.
CAG keeps the reranker fixed and adds a **non-parametric reputation prior** over documents that
updates in O(1) per feedback event, with explicit **staleness decay** and **noise/sycophancy
safeguards** — and it characterizes its own **boundary condition** (no recurrence → a strong
cross-encoder wins), which most of these do not.

---

## 3. Core statistical lineage (the math is not new — by design)

- **Beta Reputation System** (Jøsang & Ismail, 2002) — Beta-Bernoulli reputation from
  positive/negative feedback counts. CAG's α/β counters are this, per document.
- **Click models with Beta posteriors** — maintaining Beta posteriors per document to represent
  relevance/attractiveness uncertainty is standard in IR.
- **Online learning-to-rank with bandit feedback** (e.g. cascading/position-based bandits;
  LinUCB / Thompson-sampling rankers) — CAG's Thompson-sampling exploration is from this line.

**How CAG differs.** The contribution is *not* the estimator. It is the **integration**: a
clean, drop-in reputation layer that combines Beta reputation + similarity + decay + a
liar/deception counter, evaluated honestly with a stated failure mode.

---

## 4. Evaluation methodology

- **"Benchmarking is Broken — Don't Let AI Be Its Own Judge"** ([arXiv:2510.07575](https://arxiv.org/html/2510.07575v2)) —
  warns against LLM-judge circularity. CAG's evaluation independently arrived at this: an
  earlier circular Gate A (training on the eval label) was identified and replaced with an
  independent verifier, and the LLM judge is deliberately demoted.

---

## Honest gaps relative to this work

- **Not yet evaluated on Evo-Memory** (the standard streaming benchmark). Current evidence is
  on controlled sims (Gates A/B), a synthetic-distractor recurrence demo (Gate D), a
  no-recurrence boundary on real HumanEval (Gate C), and a realistic MBPP recurrence harness
  (`run_gate_recurring.py`, runnable but not yet swept on real LLM at scale).
- **Query-conditional reputation** — the per-experience granularity that ExpRAG/ReMem use is,
  in CAG, only the (unvalidated, future-work) clustering variant; validated results use global
  counters.
- **No strong-stack comparison** (hybrid retrieval + query rewriting + multi-query + agent
  memory) yet — retriever-level comparisons only.

## One-line positioning

> CAG is a lightweight, auditable **retrieval reputation layer** — Beta-Bernoulli document
> reputation with decay and sycophancy safeguards — that complements (rather than competes
> with) heavier experience-reuse memory systems, and helps specifically under **query
> recurrence with a trustworthy verifier**.
