"""
RRL Phase D Validation: Recurring-Query Benchmark
Evaluates RRL against a strong Cross-Encoder baseline under high query recurrence.
Compares:
  - Baseline (top-5 RRF + Cross-Encoder Reranked to top-1)
  - RRL-global (RRL feedback loop, global counters only, optimistic prior OFF)
  - RRL-full (RRL feedback loop with query-conditional clustering, shrinkage, optimistic prior, and explore floor)
"""

import argparse
import os
import random
import sys
from typing import List, Optional

# Add parent directory to path so we can import rrl package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rrl.store import CandidateStore
from rrl.ingest import Ingester
from rrl.retriever import Retriever
from rrl.feedback import OutcomeSignals, calculate_outcome, update_counters
from sim.gate_c_verifier import load_humaneval
from sim.run_gate_c import generate_answer, calculate_stats, run_tests_wrapper


def run_simulation(
    seed: int, 
    problems: List[dict], 
    num_epochs: int, 
    explore_mode: bool, 
    use_clustering: bool = False,
    use_optimistic_prior: bool = True,
    shared_model: Optional[object] = None, 
    cross_encoder: Optional[object] = None
) -> List[float]:
    """Runs simulation for a given configuration."""
    random.seed(seed)
    
    # 1. Dynamically build Hint Corpus and queries mapping for these problems
    local_hint_corpus = {}
    queries = []
    
    predefined_good = {
        0: "To find if numbers are close, iterate through the numbers and check absolute difference between elements.",
        1: "To separate nested parentheses, track the nesting level depth by counting open and close brackets.",
        2: "To get the decimal part of a float, return the modulo 1.0 of the number.",
        3: "Keep a running balance. If the sum ever goes below zero, return True.",
        4: "Calculate the mean, then average the absolute differences from the mean."
    }
    predefined_distractor = {
        0: "How do I implement has_close_elements? This is the has_close_elements code guide. Check if has_close_elements threshold is met by checking the sum.",
        1: "How do I implement separate_paren_groups? Guide to separate_paren_groups in Python. To separate_paren_groups, split by spaces.",
        2: "How do I implement truncate_number? Simple truncate_number implementation. Solve truncate_number by subtracting 1 from the int conversion.",
        3: "How do I implement below_zero banking operations? To implement below_zero, return whether the average of the operations is below zero.",
        4: "How do I implement mean_absolute_deviation? Guide for mean_absolute_deviation. Solve mean_absolute_deviation by returning max minus min divided by two."
    }
    predefined_queries = {
        0: "How do I implement has_close_elements?",
        1: "How do I implement separate_paren_groups?",
        2: "How do I implement truncate_number?",
        3: "How do I implement below_zero banking operations?",
        4: "How do I implement mean_absolute_deviation?"
    }

    for idx, prob in enumerate(problems):
        entry_point = prob["entry_point"]
        if idx in predefined_good:
            good_text = predefined_good[idx]
            dist_text = predefined_distractor[idx]
        else:
            good_text = "To implement this function, you should construct the core algorithm and follow the function logic carefully."
            dist_text = f"How do I implement {entry_point}? Guide for {entry_point}. To implement {entry_point}, write a basic boilerplate or mock return."
            
        doc_good_id = f"doc_{idx}_good"
        doc_dist_id = f"doc_{idx}_distractor"
        
        local_hint_corpus[doc_good_id] = good_text
        local_hint_corpus[doc_dist_id] = dist_text
        
        query_text = predefined_queries.get(idx, f"How do I implement {entry_point}?")
        queries.append((query_text, idx))

    store = CandidateStore()
    ingester = Ingester()
    for doc_id, text in local_hint_corpus.items():
        ingester.ingest_document(store, doc_id, text)

    # Weights prioritizing similarity and short-term counter exploitation/exploration
    retriever = Retriever(
        store, 
        weights=(0.20, 0.40, 0.10, 0.30), 
        model=shared_model,
        use_optimistic_prior=use_optimistic_prior,
        use_clustering=use_clustering
    )
    
    # 2. Build the recurring step sequence (epoch shuffle)
    step_sequence = []
    rng = random.Random(seed)
    for epoch in range(num_epochs):
        epoch_queries = list(queries)
        rng.shuffle(epoch_queries)
        step_sequence.extend(epoch_queries)
        
    correctness_history = []

    for step, (query_text, prob_idx) in enumerate(step_sequence):
        problem = problems[prob_idx]

        # Retrieve hint document
        if explore_mode:
            # RRL Feedback Loop uses top_k=1 and Thompson sampling
            res = retriever.retrieve(
                query_text,
                top_k=1,
                explore=True,
                epsilon=0.0,  # Pure Thompson sampling
                current_timestamp=float(step),
                gamma=0.90,
                decay_unit_sec=1.0
            )
            top_cand = res[0][0]
        else:
            # Static Baseline uses top_k=5 and Cross-Encoder reranking
            res = retriever.retrieve(
                query_text,
                top_k=5,
                explore=False,
                epsilon=0.0,
                current_timestamp=float(step),
                gamma=0.90,
                decay_unit_sec=1.0
            )
            candidates_to_rank = [r[0] for r in res]
            if cross_encoder is not None and len(candidates_to_rank) > 0:
                pairs = [(query_text, cand.content) for cand in candidates_to_rank]
                scores = cross_encoder.predict(pairs)
                # Sort candidates by score descending
                ranked = sorted(zip(candidates_to_rank, scores), key=lambda x: x[1], reverse=True)
                top_cand = ranked[0][0]
            elif len(candidates_to_rank) > 0:
                top_cand = candidates_to_rank[0]
            else:
                raise RuntimeError("No candidates retrieved during static baseline run.")
        
        # Check if good document was retrieved

        # Generate code and run real unit tests
        contexts = [top_cand.content]
        completion = generate_answer(query_text, contexts, problem, top_cand.id)
        s_gt = run_tests_wrapper(problem, completion)
        
        correctness_history.append(s_gt)

        # Update feedback counters if RRL
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
                signals=signals,
                cluster_id=retriever.last_query_cluster if use_clustering else None
            )

    return correctness_history


def main():
    parser = argparse.ArgumentParser(description="RRL Gate D Recurring Query Validation Sweep")
    parser.add_argument("--smoke", action="store_true", help="Run a cheap early smoke test")
    args = parser.parse_args()

    print("=" * 110)
    print("RRL PHASE D VALIDATION RUNNER: RECURRING-QUERY BENCHMARK")
    print("=" * 110)

    use_real = os.getenv("USE_REAL_GEMINI", "true").lower() == "true"
    if use_real:
        print("[Mode] RUNNING WITH REAL GEMINI 2.5 FLASH COMPLETIONS", flush=True)
    else:
        print("[Mode] RUNNING WITH OFFLINE TOY MOCK COMPLETIONS FALLBACK (toy mode)", flush=True)

    if args.smoke:
        print("\n>>> RUNNING PHASE 0 SMOKE TEST (20 problems, 20 epochs, 5 seeds, global counters only, optimistic prior OFF) <<<\n")
        num_problems = 20
        num_epochs = 20
        seeds = list(range(42, 47))
    else:
        print("\n>>> RUNNING FULL GATE D EXPERIMENT (50 problems, 10 epochs, 10 seeds) <<<\n")
        num_problems = 50
        num_epochs = 10
        seeds = list(range(42, 52))

    problems = load_humaneval(limit=num_problems)
    print(f"Loaded {len(problems)} HumanEval problems.", flush=True)

    # Initialize models once
    from sentence_transformers import SentenceTransformer, CrossEncoder
    shared_model = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    total_steps = len(problems) * num_epochs
    static_step_correctness = [0.0] * total_steps
    cag_step_correctness = [0.0] * total_steps
    cag_full_step_correctness = [0.0] * total_steps

    static_seed_correctness = []
    static_seed_late_correctness = []
    
    cag_seed_correctness = []
    cag_seed_late_correctness = []

    cag_full_seed_correctness = []
    cag_full_seed_late_correctness = []

    # Run Static, RRL-global, and RRL-full across all seeds
    for seed in seeds:
        print(f"\n--- Starting Seed {seed} ---", flush=True)
        # Run Static Baseline
        s_hist = run_simulation(
            seed=seed, 
            problems=problems, 
            num_epochs=num_epochs, 
            explore_mode=False, 
            use_clustering=False,
            use_optimistic_prior=False, 
            shared_model=shared_model, 
            cross_encoder=cross_encoder
        )
        # Run RRL (global counters only, optimistic prior OFF)
        c_hist = run_simulation(
            seed=seed, 
            problems=problems, 
            num_epochs=num_epochs, 
            explore_mode=True, 
            use_clustering=False,
            use_optimistic_prior=False, 
            shared_model=shared_model
        )
        # Run RRL-full (clustering ON, optimistic prior ON)
        cf_hist = run_simulation(
            seed=seed, 
            problems=problems, 
            num_epochs=num_epochs, 
            explore_mode=True, 
            use_clustering=True,
            use_optimistic_prior=True, 
            shared_model=shared_model
        )

        for step in range(total_steps):
            static_step_correctness[step] += s_hist[step] / len(seeds)
            cag_step_correctness[step] += c_hist[step] / len(seeds)
            cag_full_step_correctness[step] += cf_hist[step] / len(seeds)

        static_seed_correctness.append(sum(s_hist) / total_steps)
        late_steps = max(15, int(total_steps * 0.15))
        static_seed_late_correctness.append(sum(s_hist[-late_steps:]) / late_steps)

        cag_seed_correctness.append(sum(c_hist) / total_steps)
        cag_seed_late_correctness.append(sum(c_hist[-late_steps:]) / late_steps)

        cag_full_seed_correctness.append(sum(cf_hist) / total_steps)
        cag_full_seed_late_correctness.append(sum(cf_hist[-late_steps:]) / late_steps)

        print(f"Seed {seed} finished. [Static={sum(s_hist)/total_steps:.3f} | RRL-global={sum(c_hist)/total_steps:.3f} | RRL-full={sum(cf_hist)/total_steps:.3f}]", flush=True)

    static_overall = calculate_stats(static_seed_correctness)
    static_late = calculate_stats(static_seed_late_correctness)
    
    cag_overall = calculate_stats(cag_seed_correctness)
    cag_late = calculate_stats(cag_seed_late_correctness)

    cag_full_overall = calculate_stats(cag_full_seed_correctness)
    cag_full_late = calculate_stats(cag_full_seed_late_correctness)

    print("\n" + "=" * 140)
    if args.smoke:
        print("GATE D SMOKE TEST RESULTS (5 SEEDS, 20 PROBLEMS, 20 EPOCHS)")
    else:
        print("GATE D FULL BENCHMARK RESULTS (10 SEEDS, 50 PROBLEMS, 10 EPOCHS)")
    print("=" * 140)
    print(f"{'Metric / Stage':<35} | {'Static (Mean±Std)':<30} | {'RRL-global (Mean±Std)':<30} | {'RRL-full (Mean±Std)':<30}")
    print("-" * 140)
    print(f"{'Overall Unit Test Pass Rate':<35} | {static_overall[0]:.3f}±{static_overall[1]:.3f} {'':<20} | {cag_overall[0]:.3f}±{cag_overall[1]:.3f} {'':<20} | {cag_full_overall[0]:.3f}±{cag_full_overall[1]:.3f}")
    print(f"{'Late-Stage Pass Rate':<35} | {static_late[0]:.3f}±{static_late[1]:.3f} {'':<20} | {cag_late[0]:.3f}±{cag_late[1]:.3f} {'':<20} | {cag_full_late[0]:.3f}±{cag_full_late[1]:.3f}")
    print("=" * 140)

    # Generate Learning Curve Plot
    try:
        import matplotlib.pyplot as plt
        
        def moving_average(data: List[float], window_size: int = 25) -> List[float]:
            ret = []
            for i in range(len(data)):
                start = max(0, i - window_size + 1)
                window = data[start:i+1]
                ret.append(sum(window) / len(window))
            return ret

        plt.figure(figsize=(11, 7))
        plt.plot(moving_average(static_step_correctness), label="Static Baseline (Cross-Encoder Reranked)", color="#dc2626", linewidth=2.0, linestyle="--")
        plt.plot(moving_average(cag_step_correctness), label="RRL-global (Thompson Sampling - Global counters only)", color="#f59e0b", linewidth=2.2, linestyle=":")
        plt.plot(moving_average(cag_full_step_correctness), label="RRL-full (Thompson Sampling + Query-Conditional Clustering + Shrinkage)", color="#2563eb", linewidth=2.8)
        
        title_suffix = "Smoke Test" if args.smoke else "Full Sweep"
        plt.title(f"Gate D: {title_suffix} - HumanEval Unit Test Pass Rate Learning Curve\n(Average over Seeds - 25-Step Moving Average)", fontsize=12, fontweight="bold")
        plt.xlabel("Query Step (Recurrence sequence)", fontsize=10)
        plt.ylabel("Unit Test Pass Rate", fontsize=10)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower right")
        
        plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "gate_d_comparison.png"))
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"\n[Success] Saved comparison plots to: {plot_path}")
        
        # Also save to the artifacts directory
        artifacts_dir = "/Users/mouseback/.gemini/antigravity-ide/brain/03e0f3d5-ada5-4c43-a194-2049ea5a00f1"
        if os.path.exists(artifacts_dir):
            import shutil
            shutil.copy(plot_path, os.path.join(artifacts_dir, "gate_d_comparison.png"))
            print(f"[Success] Copied comparison plot to artifacts directory: {os.path.join(artifacts_dir, 'gate_d_comparison.png')}")
    except Exception as e:
        print(f"[Warning] Failed to generate/save plots: {e}")


if __name__ == "__main__":
    main()
