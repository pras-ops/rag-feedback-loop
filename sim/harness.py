"""
RAG Feedback Loop Simulation Harness (Phase 2 - Robustness)
Runs comparative simulations of:
1. Standard performance validation (Baseline vs Weak vs Balanced)
2. Noise Sweep (0% to 40% feedback noise)
3. Sycophancy Stress Test (counter drift with vs. without safeguards)

Saves plots to sim/results.png, sim/noise_sweep.png, and sim/counter_drift.png.
"""

import os
import random
import sys
from typing import Dict, List, Tuple

# Add parent directory to path so we can import rrl package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rrl.store import Candidate, CandidateStore
from rrl.retriever import Retriever
from rrl.feedback import OutcomeSignals, calculate_outcome, update_counters


# Simulation Configuration
NUM_STEPS = 700
TOP_K = 2
DECAY_UNIT_STEPS = 1.0
GAMMA = 1.0  # Reset decay to 1.0 for micro-simulation convergence (decay is lazy in production)

# Candidate definitions: (ID, Content, hidden_utility, base_similarity)
CANDIDATE_DEFS = [
    ("c0", "Hidden Gem (Low base relevance, high utility)", 0.95, 0.20),
    ("c1", "Solid Reference (Medium relevance, high utility)", 0.85, 0.50),
    ("c2", "Standard Doc (High relevance, medium utility)", 0.75, 0.70),
    ("c3", "Fool's Gold (Very high relevance, low utility)", 0.15, 0.90),
    ("c4", "Average Info Doc 1", 0.50, 0.40),
    ("c5", "Average Info Doc 2", 0.45, 0.45),
    ("c6", "Average Info Doc 3", 0.55, 0.35),
    ("c7", "Outdated Doc 1", 0.30, 0.60),
    ("c8", "Outdated Doc 2", 0.25, 0.50),
    ("c9", "Irrelevant Noise", 0.05, 0.10),
]


def setup_store() -> CandidateStore:
    """Initializes the store with the simulated candidates."""
    store = CandidateStore()
    for cid, content, utility, base_sim in CANDIDATE_DEFS:
        store.add_candidate(
            Candidate(
                id=cid,
                content=content,
                metadata={"hidden_utility": utility, "base_sim": base_sim},
                alpha=1.0,
                beta=1.0,
                A=1.0,
                B=1.0,
            )
        )
    return store


def generate_query_similarities(step: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Generates simulated vector and BM25 scores for a query step.
    Introduces some query-specific noise around the base similarities.
    """
    vector_scores = {}
    bm25_scores = {}
    for cid, _, _, base_sim in CANDIDATE_DEFS:
        # Add some random query variation (noise) to similarity scores
        noise_v = random.gauss(0, 0.08)
        noise_b = random.gauss(0, 0.08)
        
        # Keep scores bounded in [0.01, 0.99]
        vector_scores[cid] = max(0.01, min(0.99, base_sim + noise_v))
        bm25_scores[cid] = max(0.01, min(0.99, base_sim + noise_b))
        
    return vector_scores, bm25_scores


def simulate_user_feedback(
    utility: float,
    noise_level: float = 0.0,
    sycophancy_prob: float = 0.0
) -> OutcomeSignals:
    """
    Simulates outcome signals based on candidate's hidden ground-truth utility.
    Supports injecting random noise and sycophantic behavior.
    """
    # 1. Simulate Sycophancy
    is_sycophantic = random.random() < sycophancy_prob

    if is_sycophantic:
        # Sycophantic behavior: user always accepts and gives thumbs up
        s_behave = 0.90  # raw keep/accept signal
        s_gt = None  # verifier is independent of user sycophancy
        s_judge = max(0.0, min(1.0, random.gauss(utility, 0.15)))  # judge is independent
        s_expl = 1.0  # positive thumbs
    else:
        # Standard behavioral mapping
        r = random.random()
        if r < utility:
            s_behave = 0.90  # raw Keep/Accept
        elif r < utility + 0.5 * (1.0 - utility):
            s_behave = 0.50  # Minor Edit
        else:
            s_behave = 0.10  # Regen/Discard

        # Ground-truth verifier (exists with 60% probability)
        s_gt = None
        if random.random() < 0.60:
            s_gt = 1.0 if random.random() < utility else 0.0

        # LLM Judge signal (slightly noisy utility)
        s_judge = max(0.0, min(1.0, random.gauss(utility, 0.15)))

        # Explicit Feedback (exists 30% of the time)
        s_expl = None
        if random.random() < 0.30:
            s_expl = 1.0 if random.random() < utility else 0.0

    # 2. Inject Feedback Noise (corrupts signals to complete randomness)
    if random.random() < noise_level:
        s_behave = random.choice([0.90, 0.50, 0.10])
        if s_gt is not None:
            s_gt = random.choice([0.0, 1.0])
        s_judge = random.random()
        if s_expl is not None:
            s_expl = random.choice([0.0, 1.0])

    return OutcomeSignals(s_behave=s_behave, s_gt=s_gt, s_judge=s_judge, s_expl=s_expl)


def run_simulation(
    explore: bool,
    weights: Tuple[float, float, float, float],
    noise_level: float = 0.0,
    sycophancy_prob: float = 0.0,
    cap_behave: bool = True,
    gt_override: bool = True,
    seed: int = 42
) -> Tuple[List[float], Dict[str, List[float]], Dict[str, Candidate]]:
    """
    Runs a single simulation run.
    """
    random.seed(seed)
    
    store = setup_store()
    retriever = Retriever(store, weights=weights)
    
    retrieved_utilities = []
    c_histories = {cid: [] for cid, _, _, _ in CANDIDATE_DEFS}
    
    current_time = 0.0

    for step in range(NUM_STEPS):
        # 1. Record current C(i) values for logging
        for cid in c_histories:
            cand = store.get_candidate(cid)
            c_histories[cid].append(cand.alpha / (cand.alpha + cand.beta))
            
        # 2. Generate similarities for this query
        vector_scores, bm25_scores = generate_query_similarities(step)
        
        # 3. Retrieve candidates
        results = retriever.retrieve(
            vector_scores=vector_scores,
            bm25_scores=bm25_scores,
            top_k=TOP_K,
            explore=explore,
            epsilon=0.30 if explore else 0.0
        )
        
        # Record average utility of retrieved documents in this step
        step_utility = sum(r[0].metadata["hidden_utility"] for r in results) / len(results)
        retrieved_utilities.append(step_utility)
        
        # 4. Simulate user feedback and update counters
        retrieved_sims = {r[0].id: r[2] for r in results}
        
        # Gather signals and calculate joint outcome y
        total_y = 0.0
        for cand, _, _ in results:
            true_util = cand.metadata["hidden_utility"]
            signals = simulate_user_feedback(
                utility=true_util,
                noise_level=noise_level,
                sycophancy_prob=sycophancy_prob
            )
            y = calculate_outcome(signals, cap_behave=cap_behave, gt_override=gt_override)
            total_y += y if y is not None else 0.5
            
        avg_y = total_y / len(results)
        
        # Update store counters with decay and credit smoothing
        current_time += DECAY_UNIT_STEPS
        update_counters(
            store=store,
            retrieved_sims=retrieved_sims,
            y=avg_y,
            current_timestamp=current_time,
            gamma=GAMMA,
            decay_unit_sec=DECAY_UNIT_STEPS,
            credit_smoothing=0.50
        )

    return retrieved_utilities, c_histories, store.candidates


def print_ascii_results(
    util_exploit: List[float],
    util_weak: List[float],
    util_balanced: List[float],
    cands_exploit: Dict[str, Candidate],
    cands_weak: Dict[str, Candidate],
    cands_balanced: Dict[str, Candidate]
) -> None:
    """Prints a beautiful summary of the simulation results in the terminal."""
    print("=" * 80)
    print("SIMULATION RESULTS COMPARISON SUMMARY")
    print("=" * 80)
    
    # Calculate overall metrics
    def calc_stats(data: List[float]) -> Tuple[float, float]:
        avg_overall = sum(data) / len(data)
        avg_late = sum(data[-50:]) / 50.0
        return avg_overall, avg_late

    stats_exploit = calc_stats(util_exploit)
    stats_weak = calc_stats(util_weak)
    stats_balanced = calc_stats(util_balanced)

    print(f"{'Metric':<30} | {'Baseline (Exploit)':<18} | {'Weak Exp (70/20/10/0)':<20} | {'Balanced Exp (20/40/10/30)':<22}")
    print("-" * 98)
    print(f"{'Overall Avg Utility':<30} | {stats_exploit[0]:.4f}             | {stats_weak[0]:.4f}               | {stats_balanced[0]:.4f}")
    print(f"{'Late Stage (Last 50) Avg':<30} | {stats_exploit[1]:.4f}             | {stats_weak[1]:.4f}               | {stats_balanced[1]:.4f}")
    print("-" * 98)
    
    print("\nCandidate Convergence Analysis (True utility vs Learned expectations C(i)):")
    print(f"{'Candidate ID / Title':<38} | {'True U':<6} | {'Exploit C(i)':<12} | {'Weak Exp C(i)':<13} | {'Balanced Exp C(i)':<16}")
    print("-" * 98)
    
    for cid, name, true_u, _ in CANDIDATE_DEFS:
        c_opt = cands_exploit[cid]
        c_opt_val = c_opt.alpha / (c_opt.alpha + c_opt.beta)
        
        c_weak = cands_weak[cid]
        c_weak_val = c_weak.alpha / (c_weak.alpha + c_weak.beta)

        c_bal = cands_balanced[cid]
        c_bal_val = c_bal.alpha / (c_bal.alpha + c_bal.beta)
        
        print(f"{cid:<3} - {name[:30]:<30} | {true_u:<6.2f} | {c_opt_val:<12.4f} | {c_weak_val:<13.4f} | {c_bal_val:<16.4f}")
    print("=" * 80)


def generate_plot(
    util_exploit: List[float],
    util_weak: List[float],
    util_balanced: List[float],
    hist_balanced: Dict[str, List[float]]
):
    """Generates and saves the performance plot."""
    try:
        # pyrefly: ignore [missing-import]
        import matplotlib.pyplot as plt
        
        def moving_average(data: List[float], window_size: int = 15) -> List[float]:
            ret = []
            for i in range(len(data)):
                start = max(0, i - window_size + 1)
                window = data[start:i+1]
                ret.append(sum(window) / len(window))
            return ret

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Plot 1: Usefulness over time
        ax1.plot(moving_average(util_exploit), label="Baseline Exploitation (70/20/10, explore=False)", color="#dc2626", linewidth=2, linestyle="--")
        ax1.plot(moving_average(util_weak), label="Weak Exploration (70/20/10, explore=True)", color="#f59e0b", linewidth=2, linestyle=":")
        ax1.plot(moving_average(util_balanced), label="Balanced Exploration (20/40/10/30, explore=True)", color="#2563eb", linewidth=3)
        
        ax1.set_title("Average Retrieved Utility over Time\n(15-Step Moving Average)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Query Step", fontsize=10)
        ax1.set_ylabel("Average Utility of Top-K Candidates", fontsize=10)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="lower right")

        # Plot 2: Evolution of C(i) for key candidates
        ax2.plot(hist_balanced["c0"], label="c0 - Hidden Gem (True U = 0.95, base sim = 0.2)", color="#10b981", linewidth=2.5)
        ax2.plot(hist_balanced["c3"], label="c3 - Fool's Gold (True U = 0.15, base sim = 0.9)", color="#ef4444", linewidth=2.5)
        ax2.plot(hist_balanced["c2"], label="c2 - Standard Doc (True U = 0.75, base sim = 0.7)", color="#6366f1", linewidth=2)
        ax2.plot(hist_balanced["c1"], label="c1 - Solid Reference (True U = 0.85, base sim = 0.5)", color="#a855f7", linewidth=2)
        
        ax2.set_title("Learned Value Expectation C(i) Over Time\n(Balanced Exploration 20/40/10/30)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Query Step", fontsize=10)
        ax2.set_ylabel("C(i) = alpha / (alpha + beta)", fontsize=10)
        ax2.set_ylim(0.0, 1.0)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="best")

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
        plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "results.png"))
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[Success] Saved performance plot to: {plot_path}")
        
    except ImportError:
        print("[Warning] Matplotlib not found. Skipping plot generation.")


def run_noise_sweep_simulations():
    """
    Runs noise sweep: sweeps feedback noise level 0% to 40%.
    Plots Retrieved Utility vs Noise level.
    """
    print("\n" + "=" * 80)
    print("RUNNING FEEDBACK NOISE SWEEP (0% to 40%)")
    print("=" * 80)
    
    noise_levels = [0.0, 0.10, 0.20, 0.30, 0.40]
    seeds = [42, 43, 44]
    
    balanced_results = []
    exploit_results = []

    for noise in noise_levels:
        balanced_run_utils = []
        exploit_run_utils = []
        for seed in seeds:
            # Balanced Exploration weights (0.20, 0.40, 0.10, 0.30)
            u_bal, _, _ = run_simulation(explore=True, weights=(0.20, 0.40, 0.10, 0.30), noise_level=noise, seed=seed)
            # Baseline Exploitation weights (0.70, 0.20, 0.10, 0.0)
            u_exp, _, _ = run_simulation(explore=False, weights=(0.70, 0.20, 0.10, 0.0), noise_level=noise, seed=seed)
            
            # Record average late stage usefulness (last 100 steps)
            balanced_run_utils.append(sum(u_bal[-100:]) / 100.0)
            exploit_run_utils.append(sum(u_exp[-100:]) / 100.0)
            
        avg_bal = sum(balanced_run_utils) / len(seeds)
        avg_exp = sum(exploit_run_utils) / len(seeds)
        
        balanced_results.append(avg_bal)
        exploit_results.append(avg_exp)
        
        print(f"Noise Level: {noise * 100:.0f}% | Avg Late-Stage Utility - Balanced: {avg_bal:.4f} | Exploit: {avg_exp:.4f}")

    # Generate Plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot([n * 100 for n in noise_levels], balanced_results, marker="o", label="Balanced Exploration (20/40/10/30)", color="#2563eb", linewidth=2.5)
        plt.plot([n * 100 for n in noise_levels], exploit_results, marker="s", label="Baseline Exploitation", color="#dc2626", linewidth=2, linestyle="--")
        plt.title("Noise Tolerance Analysis\n(Late-Stage Retrieved Utility vs. Feedback Noise)", fontsize=12, fontweight="bold")
        plt.xlabel("Feedback Noise Level (%)", fontsize=10)
        plt.ylabel("Late-Stage Average Retrieved Utility", fontsize=10)
        plt.ylim(0.3, 0.7)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="best")
        
        plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "noise_sweep.png"))
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[Success] Saved noise sweep plot to: {plot_path}")
    except ImportError:
        print("[Warning] Matplotlib not found. Skipping noise sweep plotting.")


def run_sycophancy_drift_simulations():
    """
    Runs sycophancy drift stress test: 25% sycophancy probability.
    Compares the impact of different safeguards on suppressing upward drift of
    the expectation C(c3) for c3 (Fool's Gold, utility 0.15, sim 0.90).
    """
    print("\n" + "=" * 80)
    print("RUNNING SYCOPHANCY STRESS TEST (25% sycophancy, c3 Fool's Gold drift)")
    print("=" * 80)

    # 1. Run with both safeguards
    _, hist_both, final_both = run_simulation(
        explore=True,
        weights=(0.20, 0.40, 0.10, 0.30),
        sycophancy_prob=0.25,
        cap_behave=True,
        gt_override=True,
        seed=42
    )

    # 2. Run with cap_behave only (GT override disabled)
    _, hist_cap, final_cap = run_simulation(
        explore=True,
        weights=(0.20, 0.40, 0.10, 0.30),
        sycophancy_prob=0.25,
        cap_behave=True,
        gt_override=False,
        seed=42
    )

    # 3. Run with gt_override only (Cap behave disabled)
    _, hist_gt, final_gt = run_simulation(
        explore=True,
        weights=(0.20, 0.40, 0.10, 0.30),
        sycophancy_prob=0.25,
        cap_behave=False,
        gt_override=True,
        seed=42
    )

    # 4. Run with no safeguards
    _, hist_none, final_none = run_simulation(
        explore=True,
        weights=(0.20, 0.40, 0.10, 0.30),
        sycophancy_prob=0.25,
        cap_behave=False,
        gt_override=False,
        seed=42
    )

    c3_both_final = final_both["c3"].alpha / (final_both["c3"].alpha + final_both["c3"].beta)
    c3_cap_final = final_cap["c3"].alpha / (final_cap["c3"].alpha + final_cap["c3"].beta)
    c3_gt_final = final_gt["c3"].alpha / (final_gt["c3"].alpha + final_gt["c3"].beta)
    c3_none_final = final_none["c3"].alpha / (final_none["c3"].alpha + final_none["c3"].beta)

    print(f"Final C(c3) with Both Safeguards:   {c3_both_final:.4f} (Fully Suppressed)")
    print(f"Final C(c3) with Cap Behave Only:    {c3_cap_final:.4f}")
    print(f"Final C(c3) with GT Override Only:   {c3_gt_final:.4f}")
    print(f"Final C(c3) with No Safeguards:      {c3_none_final:.4f} (Upward Drift)")

    # Generate Plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 5))
        plt.plot(hist_both["c3"], label="Both Safeguards (Asymmetry + GT Override)", color="#10b981", linewidth=2.5)
        plt.plot(hist_cap["c3"], label="Cap Behave Only (Asymmetry)", color="#3b82f6", linewidth=2, linestyle="-.")
        plt.plot(hist_gt["c3"], label="GT Override Only (Verifier)", color="#a855f7", linewidth=2, linestyle=":")
        plt.plot(hist_none["c3"], label="No Safeguards (Symmetric + Blended)", color="#ef4444", linewidth=2, linestyle="--")
        plt.axhline(0.15, color="black", linestyle=":", alpha=0.5, label="True Utility (0.15)")
        
        plt.title("Sycophancy Mitigation Analysis\n(Value Expectation C(i) of Fool's Gold under 25% Sycophancy)", fontsize=12, fontweight="bold")
        plt.xlabel("Query Step", fontsize=10)
        plt.ylabel("Learned Expectation C(c3) = alpha / (alpha + beta)", fontsize=10)
        plt.ylim(0.0, 1.0)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="best")
        
        plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "counter_drift.png"))
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[Success] Saved counter drift plot to: {plot_path}")
    except ImportError:
        print("[Warning] Matplotlib not found. Skipping counter drift plotting.")


if __name__ == "__main__":
    print("Running standard Phase 1 simulations...")
    util_exploit, hist_exploit, final_exploit = run_simulation(explore=False, weights=(0.70, 0.20, 0.10, 0.0))
    util_weak, hist_weak, final_weak = run_simulation(explore=True, weights=(0.70, 0.20, 0.10, 0.0))
    util_balanced, hist_balanced, final_balanced = run_simulation(explore=True, weights=(0.20, 0.40, 0.10, 0.30))
    
    print_ascii_results(util_exploit, util_weak, util_balanced, final_exploit, final_weak, final_balanced)
    generate_plot(util_exploit, util_weak, util_balanced, hist_balanced)

    # Run Phase 2 sweeps
    run_noise_sweep_simulations()
    run_sycophancy_drift_simulations()
