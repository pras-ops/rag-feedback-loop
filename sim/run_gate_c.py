"""
CAG Phase C Validation: Real HumanEval Showcase (10-Seed Sweep)
Ingests coding hint documents (good vs distractor) for 5 HumanEval problems.
Static similarity-only retrieval gets fooled by query-dense distractor hints.
CAG feedback loop learns from unit-test results (s_gt) to suppress distractors.
Saves comparison plots to sim/gate_c_comparison.png.
"""

import math
import os
import random
import re
import signal
import subprocess
import sys
import time
from typing import List, Dict, Tuple, Optional

# Add parent directory to path so we can import cag package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cag.store import Candidate, CandidateStore
from cag.ingest import Ingester
from cag.retriever import Retriever
from cag.feedback import OutcomeSignals, calculate_outcome, update_counters
from cag.judge import _get_client
from sim.gate_c_verifier import load_humaneval, run_tests

try:
    from google.genai import types
except ImportError:
    types = None

# 1. Define Hint Corpus with Good vs. Distractor Hints
HINT_CORPUS = {
    # HumanEval/0: has_close_elements
    "doc_0_good": "To find if numbers are close, iterate through the numbers and check absolute difference between elements.",
    "doc_0_distractor": "How do I implement has_close_elements? This is the has_close_elements code guide. Check if has_close_elements threshold is met by checking the sum.",
    
    # HumanEval/1: separate_paren_groups
    "doc_1_good": "To separate nested parentheses, track the nesting level depth by counting open and close brackets.",
    "doc_1_distractor": "How do I implement separate_paren_groups? Guide to separate_paren_groups in Python. To separate_paren_groups, split by spaces.",
    
    # HumanEval/2: truncate_number
    "doc_2_good": "To get the decimal part of a float, return the modulo 1.0 of the number.",
    "doc_2_distractor": "How do I implement truncate_number? Simple truncate_number implementation. Solve truncate_number by subtracting 1 from the int conversion.",
    
    # HumanEval/3: below_zero
    "doc_3_good": "Keep a running balance. If the sum ever goes below zero, return True.",
    "doc_3_distractor": "How do I implement below_zero banking operations? To implement below_zero, return whether the average of the operations is below zero.",
    
    # HumanEval/4: mean_absolute_deviation
    "doc_4_good": "Calculate the mean, then average the absolute differences from the mean.",
    "doc_4_distractor": "How do I implement mean_absolute_deviation? Guide for mean_absolute_deviation. Solve mean_absolute_deviation by returning max minus min divided by two."
}


# Global cache to prevent redundant Gemini API calls across seeds/steps
gemini_cache: Dict[Tuple[str, str], str] = {}


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("Gemini API call timed out")


def generate_answer(query: str, contexts: List[str], problem: dict, hint_id: str) -> str:
    """Generates a solution to the coding problem, using the hint context."""
    use_real_gemini = os.getenv("USE_REAL_GEMINI", "true").lower() == "true"
    if use_real_gemini:
        client = _get_client()
        if client is None or types is None:
            raise RuntimeError(
                "Real Gemini was requested (USE_REAL_GEMINI=true) but Google GenAI Client could not be initialized.\n"
                "Please authenticate Vertex AI (GCP_PROJECT_ID) or set GEMINI_API_KEY.\n"
                "To run with the offline toy mock generator instead, run with USE_REAL_GEMINI=false."
            )
        
        cache_key = (query, hint_id)
        if cache_key in gemini_cache:
            return gemini_cache[cache_key]

        context_str = "\n".join(contexts)
        prompt = (
            f"You are a coding assistant. Complete the python function below. "
            f"Do not write markdown formatting, backticks, or comments. Just return the code. "
            f"Use the following algorithmic hint to guide your implementation:\n"
            f"Hint: {context_str}\n\n"
            f"Problem Prompt:\n{problem['prompt']}\n"
            f"Complete the function body:"
        )
        try:
            # Set per-call timeout via SIGALRM
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(30)

            print(f"[Gemini API] Requesting completion for problem '{problem.get('task_id')}' with hint '{hint_id}'...", flush=True)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                )
            )
            # Cancel alarm on success
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

            text = response.text.strip()
            print(f"[Gemini API] Success. Received response length: {len(text)} chars.", flush=True)
            
            # Clean up markdown formatting using regex
            if "```" in text:
                match = re.search(r"```(?:python)?\n?(.*?)\n?```", text, re.DOTALL)
                if match:
                    text = match.group(1)
            else:
                if text.startswith("```python"):
                    text = text[9:]
                if text.endswith("```"):
                    text = text[:-3]
            
            cleaned_text = text.strip()
            gemini_cache[cache_key] = cleaned_text
            return cleaned_text
        except _TimeoutError:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            raise RuntimeError(f"Gemini API call timed out for problem '{problem.get('task_id')}' with hint '{hint_id}'")
        except Exception as e:
            signal.alarm(0)
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass
            raise RuntimeError(f"Gemini API call failed for problem '{problem.get('task_id')}' with hint '{hint_id}': {e}")

    # Toy mock generator fallback based on correctness of retrieved hint (offline toy mode)
    context_joined = "\n".join(contexts).lower()
    good_keywords = ["absolute difference", "nesting level", "modulo 1.0", "running balance", "absolute differences"]
    is_good = any(kw in context_joined for kw in good_keywords)
    if is_good:
        return problem["canonical_solution"]
    else:
        return "    return None\n"


def run_simulation(seed: int, problems: List[dict], num_steps: int, explore_mode: bool, shared_model: Optional[object] = None) -> List[float]:
    """Runs simulation for a given configuration (explore_mode True=CAG, False=Static)."""
    random.seed(seed)
    
    store = CandidateStore()
    ingester = Ingester()
    for doc_id, text in HINT_CORPUS.items():
        ingester.ingest_document(store, doc_id, text)

    # Weights prioritizing similarity and short-term counter exploitation/exploration
    retriever = Retriever(store, weights=(0.20, 0.40, 0.10, 0.30), model=shared_model)
    
    correctness_history = []
    
    # Query prompts associated with each problem index
    queries = [
        ("How do I implement has_close_elements?", 0),
        ("How do I implement separate_paren_groups?", 1),
        ("How do I implement truncate_number?", 2),
        ("How do I implement below_zero banking operations?", 3),
        ("How do I implement mean_absolute_deviation?", 4)
    ]

    for step in range(num_steps):
        query_text, prob_idx = random.choice(queries)
        problem = problems[prob_idx]

        # Retrieve hint document
        res = retriever.retrieve(
            query_text,
            top_k=1,
            explore=explore_mode,
            epsilon=0.0,  # Pure Thompson sampling
            current_timestamp=float(step),
            gamma=0.90,
            decay_unit_sec=1.0
        )
        
        top_cand = res[0][0]
        
        # Check if good document was retrieved
        is_good_retrieved = "_good" in top_cand.id

        # Generate code and run real unit tests
        contexts = [top_cand.content]
        completion = generate_answer(query_text, contexts, problem, top_cand.id)
        s_gt = run_tests(problem, completion)
        
        correctness_history.append(s_gt)

        # Update feedback counters if CAG
        if explore_mode:
            signals = OutcomeSignals(
                s_behave=0.75 if s_gt > 0.5 else 0.10,
                s_gt=s_gt,
                s_judge=1.0 if s_gt > 0.5 else 0.0,
                s_expl=1.0 if s_gt > 0.5 else 0.0
            )
            y = calculate_outcome(signals, use_safeguards=True)
            retrieved_sims = {r[0].id: r[2] for r in res}
            update_counters(
                store=store,
                retrieved_sims=retrieved_sims,
                y=y,
                current_timestamp=float(step),
                gamma=0.90,
                decay_unit_sec=1.0,
                credit_smoothing=0.50,
                use_liar_counter=True,
                signals=signals
            )

    return correctness_history


def calculate_stats(data: List[float]) -> Tuple[float, float, float, float]:
    """Computes mean, std, and 95% Confidence Interval (using t-distribution for N=10, t=2.262)."""
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / max(1, n - 1)
    std = math.sqrt(variance)
    sem = std / math.sqrt(n)
    t_val = 2.262 if n == 10 else 1.96
    ci_half = t_val * sem
    return mean, std, mean - ci_half, mean + ci_half


def main():

    print("=" * 110)
    print("CAG PHASE C VALIDATION RUNNER: REAL HUMANEVAL UNIT-TEST VERIFIER SHOWCASE (10-SEED SWEEP)")
    print("=" * 110)

    use_real = os.getenv("USE_REAL_GEMINI", "true").lower() == "true"
    if use_real:
        print("[Mode] RUNNING WITH REAL GEMINI 2.5 FLASH COMPLETIONS (10-seed sweep)", flush=True)
    else:
        print("[Mode] RUNNING WITH OFFLINE TOY MOCK COMPLETIONS FALLBACK (toy mode)", flush=True)

    # Load HumanEval problems (first 5)
    problems = load_humaneval(limit=5)
    
    seeds = list(range(42, 52))
    num_steps = 50

    static_step_correctness = [0.0] * num_steps
    cag_step_correctness = [0.0] * num_steps

    static_seed_correctness = []
    static_seed_late_correctness = []
    
    cag_seed_correctness = []
    cag_seed_late_correctness = []

    # Initialize SentenceTransformer once to share across simulations
    from sentence_transformers import SentenceTransformer
    shared_model = SentenceTransformer("all-MiniLM-L6-v2")

    for seed in seeds:
        # Run Static (explore=False)
        s_hist = run_simulation(seed, problems, num_steps, explore_mode=False, shared_model=shared_model)
        # Run CAG (explore=True)
        c_hist = run_simulation(seed, problems, num_steps, explore_mode=True, shared_model=shared_model)

        for step in range(num_steps):
            static_step_correctness[step] += s_hist[step] / len(seeds)
            cag_step_correctness[step] += c_hist[step] / len(seeds)

        static_seed_correctness.append(sum(s_hist) / num_steps)
        static_seed_late_correctness.append(sum(s_hist[-15:]) / 15.0)

        cag_seed_correctness.append(sum(c_hist) / num_steps)
        cag_seed_late_correctness.append(sum(c_hist[-15:]) / 15.0)

        print(f"Seed {seed} finished. [Static Correctness={sum(s_hist)/num_steps:.2f} | CAG={sum(c_hist)/num_steps:.2f}]")

    # Calculate overall sweep stats
    static_overall = calculate_stats(static_seed_correctness)
    static_late = calculate_stats(static_seed_late_correctness)
    
    cag_overall = calculate_stats(cag_seed_correctness)
    cag_late = calculate_stats(cag_seed_late_correctness)

    print("\n" + "=" * 115)
    print("DECISION-GRADE GATE C RESULTS: STATIC VS CAG RETRIEVER (10-SEED SWEEP, REAL UNIT TESTS)")
    print("=" * 115)
    print(f"{'Metric / Stage':<35} | {'Static (Mean±Std [95% CI])':<38} | {'CAG (Mean±Std [95% CI])':<38}")
    print("-" * 115)
    print(f"{'Overall Unit Test Pass Rate':<35} | {static_overall[0]:.3f}±{static_overall[1]:.3f} [{static_overall[2]:.3f}, {static_overall[3]:.3f}] | {cag_overall[0]:.3f}±{cag_overall[1]:.3f} [{cag_overall[2]:.3f}, {cag_overall[3]:.3f}]")
    print(f"{'Late-Stage Pass Rate (Last 15)':<35} | {static_late[0]:.3f}±{static_late[1]:.3f} [{static_late[2]:.3f}, {static_late[3]:.3f}] | {cag_late[0]:.3f}±{cag_late[1]:.3f} [{cag_late[2]:.3f}, {cag_late[3]:.3f}]")
    print("=" * 115)

    # Generate Learning Curve Plot
    try:
        import matplotlib.pyplot as plt
        
        def moving_average(data: List[float], window_size: int = 5) -> List[float]:
            ret = []
            for i in range(len(data)):
                start = max(0, i - window_size + 1)
                window = data[start:i+1]
                ret.append(sum(window) / len(window))
            return ret

        plt.figure(figsize=(10, 6))
        plt.plot(moving_average(static_step_correctness), label="Static Baseline (RRF Similarity)", color="#dc2626", linewidth=2.5, linestyle="--")
        plt.plot(moving_average(cag_step_correctness), label="CAG Feedback Loop (Thompson Sampling)", color="#2563eb", linewidth=3.0)
        
        plt.title("Gate C: HumanEval Unit Test Pass Rate Learning Curve\n(10-Seed Average - 5-Step Moving Average)", fontsize=12, fontweight="bold")
        plt.xlabel("Query Step", fontsize=10)
        plt.ylabel("Unit Test Pass Rate", fontsize=10)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower right")
        
        plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "gate_c_comparison.png"))
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"\n[Success] Saved comparison plots to: {plot_path}")
    except ImportError:
        print("[Warning] Matplotlib not found. Skipping plot generation.")


if __name__ == "__main__":
    main()
