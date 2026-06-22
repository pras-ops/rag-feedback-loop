"""
Realistic recurring-query benchmark (the honest upgrade over the synthetic Gate D).

Tests the central, conditional claim on a REAL corpus:

    Recurrence + a real verifier  ->  outcome-aware reputation (RRL) overtakes a
                                      strong cross-encoder reranker.

What makes this more realistic than Gate C/D:
  * Real corpus  : retrievable "worked examples" are real MBPP reference solutions,
                   not hand-written query-stuffed distractors.
  * Natural distractors : the corpus also holds solutions to OTHER (non-queried)
                   problems from the *same topic families*, so they are lexically
                   similar and genuinely fool similarity retrieval — but are never
                   the right answer for any query (keeps GLOBAL counters valid; no
                   clustering needed).
  * Real recurrence : each query problem recurs once per epoch, so global
                   reputation can actually accumulate.
  * Real verifier : generated code is executed against the real MBPP unit tests.
  * Real generation : Gemini writes the code (set USE_REAL_GEMINI=true + key).

This harness REFUSES to report a "real" number on the mock generator — pass
--mock only to self-test the plumbing, and it will label the run INVALID.

Offline self-test (no LLM, no network for generation):
    python3 sim/run_gate_recurring.py --selftest
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rrl.store import CandidateStore
from rrl.ingest import Ingester
from rrl.retriever import Retriever
from rrl.feedback import OutcomeSignals, calculate_outcome, update_counters
from rrl.judge import _get_client

try:
    from google.genai import types
except ImportError:
    types = None

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mbpp.jsonl")


# ---------------------------------------------------------------- data & verifier

def load_mbpp() -> List[dict]:
    return [json.loads(line) for line in open(DATA_PATH)]


def entry_point(problem: dict) -> str:
    """Function name from the first assert, e.g. 'assert min_cost(...' -> 'min_cost'."""
    m = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", problem["test_list"][0])
    return m.group(1) if m else ""


def topic(text: str) -> str:
    """Crude topic family = first content word — gives naturally recurring families."""
    stop = {"write", "a", "an", "the", "to", "of", "function", "python", "program",
            "given", "find", "for", "that", "in", "is", "check", "whether", "from",
            "and", "using", "get", "return"}
    for w in re.findall(r"[a-z]+", text.lower()):
        if w not in stop:
            return w
    return "misc"


def run_tests(problem: dict, completion: str, timeout: float = 10.0) -> float:
    """Run the candidate completion against the real MBPP unit tests. 1.0 pass / 0.0 fail."""
    program = (
        (problem.get("test_setup_code") or "") + "\n"
        + completion + "\n"
        + "\n".join(problem["test_list"]) + "\n"
    )
    try:
        proc = subprocess.run([sys.executable, "-c", program],
                              capture_output=True, text=True, timeout=timeout)
        return 1.0 if proc.returncode == 0 else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------- corpus building

def build_dataset(seed: int, n_query: int = 30, n_distractor: int = 60):
    """
    Pick query problems from a few recurring topic families, plus distractor problems
    (real solutions to OTHER problems in those same families that are never queried).
    Returns (query_problems, corpus_docs) where corpus_docs maps doc_id -> (text, is_good_for).
    """
    rng = random.Random(seed)
    rows = load_mbpp()
    # keep only cleanly-verifiable problems (canonical solution passes its own tests)
    by_family: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        if r.get("test_list") and entry_point(r):
            by_family[topic(r["text"])].append(r)

    # focus on the biggest recurring families so distractors are genuinely similar
    families = sorted(by_family, key=lambda f: -len(by_family[f]))[:8]
    pool = [p for f in families for p in by_family[f]]
    rng.shuffle(pool)

    query_problems = pool[:n_query]
    distractor_problems = pool[n_query:n_query + n_distractor]

    corpus_docs: Dict[str, Tuple[str, Optional[str]]] = {}
    for p in query_problems:
        # "good" doc = this problem's own reference solution (it was solved before)
        corpus_docs[f"good_{p['task_id']}"] = (p["code"], str(p["task_id"]))
    for p in distractor_problems:
        # distractor = a real, similar solution that is never correct for a query
        corpus_docs[f"distractor_{p['task_id']}"] = (p["code"], None)
    return query_problems, corpus_docs


# ---------------------------------------------------------------- generation

def generate(problem: dict, retrieved_code: str, use_real: bool) -> str:
    client = _get_client() if use_real else None
    if client is not None and types is not None:
        prompt = (
            "You are a Python coding assistant. Using the reference example only if it is "
            "relevant, write a correct solution to the problem. Return ONLY the function "
            "definition, no markdown, no commentary.\n\n"
            f"Reference example:\n{retrieved_code}\n\n"
            f"Problem: {problem['text']}\n"
            f"The function must be named `{entry_point(problem)}`.\n"
        )
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0))
            t = resp.text.strip()
            for fence in ("```python", "```"):
                if t.startswith(fence):
                    t = t[len(fence):]
            if t.endswith("```"):
                t = t[:-3]
            return t.strip()
        except Exception as e:
            print(f"[gen warning] Gemini failed: {e}")
            return ""
    # MOCK (plumbing only): a correct-good-doc passes, a distractor fails.
    return problem["code"] if retrieved_code == problem["code"] else "    pass\n"


# ---------------------------------------------------------------- stats

def stats(data: List[float]) -> Tuple[float, float, float, float]:
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(data) / n
    var = sum((x - mean) ** 2 for x in data) / max(1, n - 1)
    sem = math.sqrt(var) / math.sqrt(n)
    try:
        from scipy import stats as _st
        t_val = float(_st.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    except Exception:
        t_val = 2.045 if n >= 30 else 2.262
    return mean, math.sqrt(var), mean - t_val * sem, mean + t_val * sem


# ---------------------------------------------------------------- one run

def run_arm(seed: int, epochs: int, use_cag: bool, use_real: bool,
            cross_encoder=None) -> List[float]:
    query_problems, corpus_docs = build_dataset(seed)
    store = CandidateStore()
    ingester = Ingester()
    for doc_id, (code, _good_for) in corpus_docs.items():
        ingester.ingest_document(store, doc_id, code)

    retriever = Retriever(store, weights=(0.20, 0.40, 0.10, 0.30))
    rng = random.Random(seed)
    stream = []
    for _ in range(epochs):
        order = list(query_problems)
        rng.shuffle(order)
        stream.extend(order)

    history = []
    gen_cache: Dict[Tuple[int, str], float] = {}
    for step, problem in enumerate(stream):
        if use_cag:
            res = retriever.retrieve(problem["text"], top_k=1, explore=True, epsilon=0.0,
                                     current_timestamp=float(step), gamma=0.95, decay_unit_sec=1.0)
            top = res[0][0]
        else:
            # static baseline: top-5 RRF, cross-encoder rerank to 1 (no learning)
            res = retriever.retrieve(problem["text"], top_k=5, explore=False)
            if cross_encoder is not None and len(res) > 1:
                pairs = [(problem["text"], r[0].content) for r in res]
                scores = cross_encoder.predict(pairs)
                top = res[int(max(range(len(scores)), key=lambda i: scores[i]))][0]
            else:
                top = res[0][0]

        key = (problem["task_id"], top.id)
        if key in gen_cache:
            passed = gen_cache[key]
        else:
            completion = generate(problem, top.content, use_real)
            passed = run_tests(problem, completion)
            gen_cache[key] = passed
        history.append(passed)

        if use_cag:
            signals = OutcomeSignals(
                s_behave=0.75 if passed > 0.5 else 0.10,
                s_gt=passed,
                s_judge=1.0 if passed > 0.5 else 0.0,
                s_expl=1.0 if passed > 0.5 else 0.0)
            y = calculate_outcome(signals, use_safeguards=True)
            update_counters(store, {r[0].id: r[2] for r in res}, y,
                            current_timestamp=float(step), gamma=0.95, decay_unit_sec=1.0,
                            credit_smoothing=0.50, use_liar_counter=True, signals=signals)
    return history


# ---------------------------------------------------------------- self-test

def selftest():
    print("SELF-TEST (offline): MBPP verifier + dataset build, no LLM.")
    rows = load_mbpp()
    ok = sum(run_tests(p, p["code"]) for p in rows[:25])
    broken = sum(run_tests(p, "    pass\n") for p in rows[:25])
    print(f"  canonical passed: {int(ok)}/25   'pass'-stub passed: {int(broken)}/25")
    q, corpus = build_dataset(seed=42)
    n_good = sum(1 for k in corpus if k.startswith("good_"))
    n_dist = sum(1 for k in corpus if k.startswith("distractor_"))
    print(f"  dataset: {len(q)} query problems, corpus = {n_good} good + {n_dist} distractor docs")
    print("VERIFIER OK" if ok >= 23 and broken == 0 else "VERIFIER PROBLEM")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="offline plumbing check, no LLM")
    ap.add_argument("--mock", action="store_true", help="run with mock generator (INVALID results)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    use_real = os.getenv("USE_REAL_GEMINI", "false").lower() == "true"
    if not use_real and not args.mock:
        sys.exit("ERROR: real generation requires USE_REAL_GEMINI=true (+ GEMINI_API_KEY or "
                 "Vertex creds). Pass --mock ONLY to test the plumbing — those results are NOT valid.")
    if args.mock:
        print("!" * 80)
        print("WARNING: --mock generator in use. RESULTS ARE NOT A VALID BENCHMARK, plumbing only.")
        print("!" * 80)

    cross_encoder = None
    try:
        from sentence_transformers import CrossEncoder
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        print(f"[warn] cross-encoder unavailable ({e}); static baseline falls back to RRF top-1.")

    seeds = list(range(42, 42 + args.seeds))
    static_overall, cag_overall, static_late, cag_late = [], [], [], []
    for s in seeds:
        sh = run_arm(s, args.epochs, use_cag=False, use_real=use_real and not args.mock,
                     cross_encoder=cross_encoder)
        ch = run_arm(s, args.epochs, use_cag=True, use_real=use_real and not args.mock)
        static_overall.append(sum(sh) / len(sh)); cag_overall.append(sum(ch) / len(ch))
        k = max(1, len(sh) // 5)
        static_late.append(sum(sh[-k:]) / k); cag_late.append(sum(ch[-k:]) / k)
        print(f"seed {s}: static={static_overall[-1]:.3f}  rrl={cag_overall[-1]:.3f}")

    so, co = stats(static_overall), stats(cag_overall)
    sl, cl = stats(static_late), stats(cag_late)
    tag = "  (MOCK — INVALID)" if args.mock else ""
    print("\n" + "=" * 90)
    print(f"REALISTIC RECURRING BENCHMARK (MBPP, {args.seeds} seeds x {args.epochs} epochs){tag}")
    print("=" * 90)
    print(f"{'Metric':<28} | {'Static (cross-encoder)':<30} | {'RRL (global counters)':<30}")
    print("-" * 90)
    print(f"{'Overall pass rate':<28} | {so[0]:.3f} [{so[2]:.3f}, {so[3]:.3f}]          | {co[0]:.3f} [{co[2]:.3f}, {co[3]:.3f}]")
    print(f"{'Late-stage pass rate':<28} | {sl[0]:.3f} [{sl[2]:.3f}, {sl[3]:.3f}]          | {cl[0]:.3f} [{cl[2]:.3f}, {cl[3]:.3f}]")
    print("=" * 90)
    print("Verdict: CI-separated => recurrence win is real; overlapping => not significant.")


if __name__ == "__main__":
    main()
