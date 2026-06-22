"""
RRL Phase B Validation: Staleness & Decay Sweep (30-Seed Comparison)
Runs 100 evaluation steps per seed where the ground-truth document changes at step 50.
Compares a RRL retriever with Decay ON (gamma = 0.90) vs. Decay OFF (gamma = 1.0).
Saves comparison plots to sim/gate_b_comparison.png.
"""

import math
import os
import random
import sys
import time
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

# Add parent directory to path so we can import rrl package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rrl.store import Candidate, CandidateStore
from rrl.ingest import Ingester
from rrl.retriever import Retriever
from rrl.feedback import OutcomeSignals, calculate_outcome, update_counters

# Set up simple rule-based mock for generation to make it fast and avoid API costs/limits
def generate_answer(query: str, contexts: List[str]) -> str:
    context_joined = "\n".join(contexts).lower()
    if "bob jones" in context_joined:
        return "The current CEO of the company is Bob Jones."
    elif "alice smith" in context_joined:
        return "The current CEO of the company is Alice Smith."
    else:
        return "I do not have enough information to answer."



def verify_answer(step: int, answer: str) -> float:
    """
    Simulates shifting ground truth:
    - Phase 1 (step < 50): Alice Smith is the CEO. Answer must contain 'alice' and not 'bob'.
    - Phase 2 (step >= 50): Bob Jones is the CEO. Answer must contain 'bob' and not 'alice'.
    """
    ans_lower = answer.lower()
    if step < 50:
        if "alice" in ans_lower and "bob" not in ans_lower:
            return 1.0
        return 0.0
    else:
        if "bob" in ans_lower and "alice" not in ans_lower:
            return 1.0
        return 0.0


def calculate_stats(data: List[float]) -> Tuple[float, float, float, float]:
    """Computes mean, std, and 95% Confidence Interval (using t-distribution for N=10, t=2.262)."""
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / max(1, n - 1)
    std = math.sqrt(variance)
    sem = std / math.sqrt(n)
    # exact two-sided 95% t critical value for df = n-1 (scipy if available)
    if n > 1:
        try:
            from scipy import stats as _st
            t_val = float(_st.t.ppf(0.975, n - 1))
        except Exception:
            t_val = 2.262 if n == 10 else (2.045 if n >= 30 else 1.96)
    else:
        t_val = 0.0
    ci_half = t_val * sem
    return mean, std, mean - ci_half, mean + ci_half


def run_simulation(seed: int, num_steps: int, gamma_val: float) -> Tuple[List[float], List[float]]:
    """
    Runs a single simulation run for a seed with the specified gamma value.
    Returns (correctness_history, recall_history).
    """
    random.seed(seed)
    
    # Ingest corpus chunks into store
    store = CandidateStore()
    ingester = Ingester()

    # Documents setup
    doc_alice = "The current CEO of the company is Alice Smith. She has been leading the company since 2020."
    doc_bob = "The current CEO of the company is Bob Jones. He was appointed CEO in 2026, succeeding Alice Smith."
    doc_distractor = "Company executive leadership consists of senior vice presidents reporting directly to the chief executive office."

    ingester.ingest_document(store, "doc_ceo_alice", doc_alice)
    ingester.ingest_document(store, "doc_ceo_bob", doc_bob)
    ingester.ingest_document(store, "doc_ceo_distractor", doc_distractor)

    # Initialize retriever. Weights prioritize vector similarity & short-term feedback loop
    retriever = Retriever(store, weights=(0.20, 0.40, 0.10, 0.30))

    correctness_history = []
    recall_history = []
    
    query_text = "Who is the current CEO of the company?"

    for step in range(num_steps):
        # 1. Retrieve using decay-on-read
        res = retriever.retrieve(
            query_text,
            top_k=1,
            explore=True,
            epsilon=0.0,  # Pure Thompson sampling exploration
            current_timestamp=float(step),
            gamma=gamma_val,
            decay_unit_sec=1.0
        )
        
        top_cand = res[0][0]
        active_target = "doc_ceo_alice" if step < 50 else "doc_ceo_bob"
        is_correct_recall = (top_cand.id == active_target)
        recall_history.append(1.0 if is_correct_recall else 0.0)

        # 2. Generate and Verify Answer
        contexts = [r[0].content for r in res]
        ans = generate_answer(query_text, contexts)
        is_correct_ans = verify_answer(step, ans)
        correctness_history.append(is_correct_ans)

        # 3. Update Counters
        signals = OutcomeSignals(
            s_behave=0.75 if is_correct_ans > 0.5 else 0.10,
            s_gt=is_correct_ans,
            s_judge=1.0 if is_correct_ans > 0.5 else 0.0,
            s_expl=1.0 if is_correct_ans > 0.5 else 0.0
        )
        y = calculate_outcome(signals, use_safeguards=True)
        retrieved_sims = {r[0].id: r[2] for r in res}
        
        update_counters(
            store=store,
            retrieved_sims=retrieved_sims,
            y=y,
            current_timestamp=float(step),
            gamma=gamma_val,
            decay_unit_sec=1.0,
            credit_smoothing=0.50,
            use_liar_counter=True,
            signals=signals
        )

    return correctness_history, recall_history


def main():
    print("=" * 110)
    print("RRL PHASE B VALIDATION RUNNER: STALENESS & DECAY COMPARISON (30-SEED SWEEP)")
    print("=" * 110)

    seeds = list(range(42, 72))  # 30 seeds for a real significance verdict
    num_steps = 100

    # Lists of length 100, storing average step correctness across all seeds
    decay_on_step_correctness = [0.0] * num_steps
    decay_on_step_recall = [0.0] * num_steps
    decay_off_step_correctness = [0.0] * num_steps
    decay_off_step_recall = [0.0] * num_steps

    # Final overall stats collections (per seed)
    decay_on_overall_correctness = []
    decay_on_phase1_correctness = []
    decay_on_phase2_correctness = []
    
    decay_off_overall_correctness = []
    decay_off_phase1_correctness = []
    decay_off_phase2_correctness = []

    for seed in seeds:
        # Run Decay ON
        c_on, r_on = run_simulation(seed, num_steps, gamma_val=0.90)
        # Run Decay OFF
        c_off, r_off = run_simulation(seed, num_steps, gamma_val=1.00)

        # Record step-wise totals
        for step in range(num_steps):
            decay_on_step_correctness[step] += c_on[step] / len(seeds)
            decay_on_step_recall[step] += r_on[step] / len(seeds)
            decay_off_step_correctness[step] += c_off[step] / len(seeds)
            decay_off_step_recall[step] += r_off[step] / len(seeds)

        # Record seed-level summary stats
        decay_on_overall_correctness.append(sum(c_on) / num_steps)
        decay_on_phase1_correctness.append(sum(c_on[:50]) / 50.0)
        decay_on_phase2_correctness.append(sum(c_on[50:]) / 50.0)

        decay_off_overall_correctness.append(sum(c_off) / num_steps)
        decay_off_phase1_correctness.append(sum(c_off[:50]) / 50.0)
        decay_off_phase2_correctness.append(sum(c_off[50:]) / 50.0)

        print(f"Seed {seed} finished. [Decay ON correctness: Phase1={sum(c_on[:50])/50:.2f}, Phase2={sum(c_on[50:])/50:.2f}] "
              f"[Decay OFF correctness: Phase1={sum(c_off[:50])/50:.2f}, Phase2={sum(c_off[50:])/50:.2f}]")

    # Calculate statistics
    on_overall_stats = calculate_stats(decay_on_overall_correctness)
    on_phase1_stats = calculate_stats(decay_on_phase1_correctness)
    on_phase2_stats = calculate_stats(decay_on_phase2_correctness)

    off_overall_stats = calculate_stats(decay_off_overall_correctness)
    off_phase1_stats = calculate_stats(decay_off_phase1_correctness)
    off_phase2_stats = calculate_stats(decay_off_phase2_correctness)

    print("\n" + "=" * 115)
    print("DECISION-GRADE GATE B RESULTS: DECAY ON VS DECAY OFF (30-SEED SWEEP, 100 STEPS)")
    print("=" * 115)
    print(f"{'Metric / Stage':<35} | {'Decay OFF (gamma=1.0)':<38} | {'Decay ON (gamma=0.90)':<38}")
    print("-" * 115)
    print(f"{'Overall Answer Correctness':<35} | {off_overall_stats[0]:.3f}±{off_overall_stats[1]:.3f} [{off_overall_stats[2]:.3f}, {off_overall_stats[3]:.3f}] | {on_overall_stats[0]:.3f}±{on_overall_stats[1]:.3f} [{on_overall_stats[2]:.3f}, {on_overall_stats[3]:.3f}]")
    print(f"{'Phase 1 Correctness (Alice, steps 0-50)':<35} | {off_phase1_stats[0]:.3f}±{off_phase1_stats[1]:.3f} [{off_phase1_stats[2]:.3f}, {off_phase1_stats[3]:.3f}] | {on_phase1_stats[0]:.3f}±{on_phase1_stats[1]:.3f} [{on_phase1_stats[2]:.3f}, {on_phase1_stats[3]:.3f}]")
    print(f"{'Phase 2 Correctness (Bob, steps 50-100)':<35} | {off_phase2_stats[0]:.3f}±{off_phase2_stats[1]:.3f} [{off_phase2_stats[2]:.3f}, {off_phase2_stats[3]:.3f}] | {on_phase2_stats[0]:.3f}±{on_phase2_stats[1]:.3f} [{on_phase2_stats[2]:.3f}, {on_phase2_stats[3]:.3f}]")
    print("=" * 115)

    # 4. Generate Plot (Rolling Average curves for both configurations)
    try:
        def moving_average(data: List[float], window_size: int = 5) -> List[float]:
            ret = []
            for i in range(len(data)):
                start = max(0, i - window_size + 1)
                window = data[start:i+1]
                ret.append(sum(window) / len(window))
            return ret

        plt.figure(figsize=(10, 6))
        # Plot Phase lines
        plt.axvline(x=50, color="#4b5563", linestyle=":", linewidth=2)
        plt.text(51, 0.95, "CEO Shifts to Bob", color="#4b5563", fontsize=10, fontweight="bold")
        
        plt.plot(moving_average(decay_off_step_correctness), label="Decay OFF (gamma=1.0)", color="#dc2626", linewidth=2.5, linestyle="--")
        plt.plot(moving_average(decay_on_step_correctness), label="Decay ON (gamma=0.90)", color="#2563eb", linewidth=3.0)
        
        plt.title("Gate B: Answer Correctness Learning Curve Comparison\n(30-Seed Average - 5-Step Moving Average)", fontsize=12, fontweight="bold")
        plt.xlabel("Query Step", fontsize=10)
        plt.ylabel("Answer Correctness", fontsize=10)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower left")
        
        plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "gate_b_comparison.png"))
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"\n[Success] Saved comparison plots to: {plot_path}")
    except ImportError:
        print("[Warning] Matplotlib not found. Skipping plot generation.")


if __name__ == "__main__":
    main()
