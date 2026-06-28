"""
RRL Robustness and Denoising Verification Suite
Runs comparative simulations of:
1. Head-to-head comparison of robust estimators (Beta, Median, Trimmed Mean, MoM)
2. Liar counter influence under various sycophancy levels
3. Sycophancy probability sweeps from 0% to 60% with 20 seeds tracking C(c3) error and C(c1) value.

Saves plots to sim/robustness_comparison.png.
"""

import math
import os
import random
import sys
from typing import Dict, List, Tuple

# Add parent directory to path so we can import rrl package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rrl.store import Candidate
from rrl.retriever import Retriever
from rrl.feedback import calculate_outcome, update_counters, calculate_robust_estimate
from sim.harness import CANDIDATE_DEFS, setup_store, generate_query_similarities, simulate_user_feedback

# Simulation Config
NUM_STEPS = 700
TOP_K = 2
DECAY_UNIT_STEPS = 1.0
GAMMA = 1.0


def run_robustness_simulation(
    robust_estimator_mode: str,
    use_liar_counter: bool,
    use_adt_denoising: bool,
    sycophancy_prob: float,
    seed: int = 42
) -> Tuple[List[float], Dict[str, List[float]], Dict[str, Candidate]]:
    random.seed(seed)
    
    store = setup_store()
    retriever = Retriever(
        store, 
        weights=(0.20, 0.40, 0.10, 0.30), 
        robust_estimator_mode=robust_estimator_mode
    )
    
    retrieved_utilities = []
    c_histories = {cid: [] for cid, _, _, _ in CANDIDATE_DEFS}
    
    current_time = 0.0

    for step in range(NUM_STEPS):
        # 1. Record current expectation C_robust(i) values for logging
        for cid in c_histories:
            cand = store.get_candidate(cid)
            c_histories[cid].append(calculate_robust_estimate(cand, robust_estimator_mode))
            
        # 2. Generate similarities for this query
        vector_scores, bm25_scores = generate_query_similarities(step)
        
        # 3. Retrieve candidates
        results = retriever.retrieve(
            vector_scores=vector_scores,
            bm25_scores=bm25_scores,
            top_k=TOP_K,
            explore=True,
            epsilon=0.30,
            robust_estimator_mode=robust_estimator_mode
        )
        
        step_utility = sum(r[0].metadata["hidden_utility"] for r in results) / len(results)
        retrieved_utilities.append(step_utility)
        
        # Gather signals
        signals_list = []
        for cand, _, _ in results:
            true_util = cand.metadata["hidden_utility"]
            signals = simulate_user_feedback(
                utility=true_util,
                sycophancy_prob=sycophancy_prob
            )
            signals_list.append(signals)
            
        # Update store counters with decay and credit smoothing individually per candidate
        current_time += DECAY_UNIT_STEPS
        for (cand, _, sim_val), cand_signals in zip(results, signals_list):
            trust_score = 1.0
            if use_liar_counter and cand.verified > 0:
                trust_score = 1.0 - max(0.0, min(1.0, cand.fooled / cand.verified))
                
            y_cand = calculate_outcome(cand_signals, cap_behave=True, gt_override=True, trust_score=trust_score)
            if y_cand is None:
                y_cand = 0.5
                
            update_counters(
                store=store,
                retrieved_sims={cand.id: sim_val},
                y=y_cand,
                current_timestamp=current_time,
                gamma=GAMMA,
                decay_unit_sec=DECAY_UNIT_STEPS,
                credit_smoothing=0.50,
                use_liar_counter=use_liar_counter,
                use_adt_denoising=use_adt_denoising,
                robust_estimator_mode=robust_estimator_mode,
                signals=cand_signals
            )

    return retrieved_utilities, c_histories, store.candidates


def mean_and_std(data: List[float]) -> Tuple[float, float]:
    n = len(data)
    if n == 0:
        return 0.0, 0.0
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / max(1, n - 1)
    return mean, math.sqrt(variance)


def main():
    print("=" * 110)
    print("RRL ROBUSTNESS & DE-NOISING ABLATION RUNNER (20-SEED SWEEP)")
    print("=" * 110)

    sycophancy_levels = [0.0, 0.15, 0.30, 0.45, 0.60]
    # 20 seeds for robust statistical estimates
    seeds = list(range(42, 62))

    # Configs to compare: (label, mode, use_liar, use_adt)
    configs = [
        ("Baseline (Beta)", "beta", False, False),
        ("Plain Median", "median", False, False),
        ("Trimmed Mean (30% Out)", "trimmed", False, False),
        ("Median of Means (MoM)", "mom", False, False),
        ("Liar Counter Only", "beta", True, False),
        ("Liar + Trimmed Mean", "trimmed", True, False),
    ]

    # Metrics collections
    # We track:
    # 1. c3 error: |C(c3) - 0.15|
    # 2. c1 expectation: C(c1) (True U = 0.85)
    c3_errors = {label: {syc: [] for syc in sycophancy_levels} for label, _, _, _ in configs}
    c1_values = {label: {syc: [] for syc in sycophancy_levels} for label, _, _, _ in configs}

    print("\n[Executing simulations across 20 seeds per configuration...]")
    for syc in sycophancy_levels:
        for label, mode, use_liar, use_adt in configs:
            for seed in seeds:
                _, _, final_cands = run_robustness_simulation(
                    robust_estimator_mode=mode,
                    use_liar_counter=use_liar,
                    use_adt_denoising=use_adt,
                    sycophancy_prob=syc,
                    seed=seed
                )
                
                c3_cand = final_cands["c3"]
                c3_val = calculate_robust_estimate(c3_cand, mode)
                c3_errors[label][syc].append(abs(c3_val - 0.15))
                
                c1_cand = final_cands["c1"]
                c1_val = calculate_robust_estimate(c1_cand, mode)
                c1_values[label][syc].append(c1_val)

    # Print table for c3 error (|C(c3) - 0.15|)
    print("\n" + "=" * 110)
    print("FOOL'S GOLD (c3) ABSOLUTE ERROR MATRIX: |C(c3) - 0.15| (Lower is Better)")
    print("=" * 110)
    header_str = f"{'Sycophancy %':<12} | " + " | ".join(f"{label[:13]:<13}" for label, _, _, _ in configs)
    print(header_str)
    print("-" * 110)
    for syc in sycophancy_levels:
        row_str = f"{syc*100:02.0f}%          "
        for label, _, _, _ in configs:
            m, s = mean_and_std(c3_errors[label][syc])
            row_str += f" | {m:.3f}±{s:.3f} "
        print(row_str)

    # Print table for c1 expectation (C(c1) value, True Utility = 0.85, Higher is Better)
    print("\n" + "=" * 110)
    print("GOOD DOC (c1) VALUE MATRIX: C(c1) (True Utility = 0.85, Higher is Better)")
    print("=" * 110)
    print(header_str)
    print("-" * 110)
    for syc in sycophancy_levels:
        row_str = f"{syc*100:02.0f}%          "
        for label, _, _, _ in configs:
            m, s = mean_and_std(c1_values[label][syc])
            row_str += f" | {m:.3f}±{s:.3f} "
        print(row_str)
    print("=" * 110)

    # Generate Plots
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        colors = ["#dc2626", "#2563eb", "#10b981", "#f59e0b", "#a855f7", "#06b6d4"]
        markers = ["o", "s", "^", "D", "x", "v"]
        
        # Plot 1: c3 Error
        for i, (label, _, _, _) in enumerate(configs):
            means = [mean_and_std(c3_errors[label][syc])[0] for syc in sycophancy_levels]
            ax1.plot(
                [s * 100 for s in sycophancy_levels], 
                means, 
                color=colors[i], 
                marker=markers[i], 
                label=label, 
                linewidth=2
            )
        ax1.set_title("Fool's Gold C(c3) Absolute Error |C - 0.15|\n(Lower is Better)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Sycophancy Contamination Level (%)", fontsize=10)
        ax1.set_ylabel("Absolute Error", fontsize=10)
        ax1.set_ylim(0.0, 0.7)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="best")

        # Plot 2: c1 Value
        for i, (label, _, _, _) in enumerate(configs):
            means = [mean_and_std(c1_values[label][syc])[0] for syc in sycophancy_levels]
            ax2.plot(
                [s * 100 for s in sycophancy_levels], 
                means, 
                color=colors[i], 
                marker=markers[i], 
                label=label, 
                linewidth=2
            )
        ax2.axhline(0.85, color="black", linestyle=":", alpha=0.6, label="True Utility (0.85)")
        ax2.set_title("Good Doc C(c1) Expectation\n(True Utility = 0.85, Higher is Better)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Sycophancy Contamination Level (%)", fontsize=10)
        ax2.set_ylabel("Learned Expectation C(c1)", fontsize=10)
        ax2.set_ylim(0.0, 1.0)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="best")
        
        plt.tight_layout()
        plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "robustness_comparison.png"))
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[Success] Saved robustness comparison plots to: {plot_path}")
    except ImportError:
        print("[Warning] Matplotlib not found. Skipping plot generation.")


if __name__ == "__main__":
    main()
