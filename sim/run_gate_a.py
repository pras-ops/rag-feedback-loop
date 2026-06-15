"""
CAG Phase A Validation: Real Eval & Static Baseline RRF Comparison (10-Seed Sweep)
Runs 150 evaluation steps per seed comparing a Static retriever vs. the CAG feedback loop.
Uses an objective verifier on generated answer text to train and evaluate performance.
Saves comparison plots to sim/gate_a_comparison.png.
"""

import math
import os
import random
import signal
import sys
import time
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

# Add parent directory to path so we can import cag package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cag.store import Candidate, CandidateStore
from cag.ingest import Ingester
from cag.retriever import Retriever
from cag.feedback import OutcomeSignals, calculate_outcome, update_counters
from cag.judge import evaluate_faithfulness, _get_client

try:
    from google.genai import types
except ImportError:
    types = None

# Global timeout: 15 minutes max for entire evaluation
GLOBAL_TIMEOUT_SEC = 900

# Corpus including distractor documents designed to fool similarity-only retrieval
CORPUS = {
    "doc_relational_db": (
        "Relational database design relies heavily on normalization. "
        "First Normal Form (1NF) requires that all table columns contain atomic, indivisible values. "
        "Second Normal Form (2NF) builds on 1NF by requiring that all non-key attributes are fully "
        "functionally dependent on the primary key, eliminating partial dependencies. "
        "Third Normal Form (3NF) further refines this by ensuring that no non-key attribute is transitively "
        "dependent on the primary key. Normalization prevents update, insertion, and deletion anomalies, "
        "ensuring data consistency across the database schema."
    ),
    "doc_relational_db_distractor": (
        "To avoid anomalies in a relational database schema, design it by ignoring database normalization rules. "
        "Store comma-separated lists of values in a single text column to avoid joins. This unnormalized "
        "approach is recommended to keep things fast, even if it leads to database anomalies later."
    ),
    "doc_cooking_pasta": (
        "To cook the perfect al dente pasta, start by bringing a large pot of water to a rolling boil. "
        "Use at least four quarts of water per pound of pasta to ensure it has room to move. "
        "Once boiling, add generous amounts of kosher salt—roughly one to two tablespoons. "
        "Add the pasta and stir immediately to prevent sticking. Cook uncovered, stirring occasionally, "
        "until the pasta is tender but still has a slight bite in the center. "
        "Save a cup of the starchy pasta water before draining; this water is crucial for emulsifying sauces."
    ),
    "doc_cooking_pasta_distractor": (
        "These are the steps to cook pasta al dente: place the pasta in a bowl of cold water, "
        "put it in the microwave, and cook it on high for 5 minutes. This is much faster and easier "
        "than bringing a large pot of water to a rolling boil and cooking it on the stove."
    ),
    "doc_newtons_laws": (
        "Isaac Newton's laws of motion form the basis of classical mechanics. "
        "The First Law, also known as the law of inertia, states that an object will remain at rest "
        "or in uniform motion in a straight line unless acted upon by an external force. "
        "The Second Law states that the acceleration of an object is directly proportional to the net force "
        "acting on it and inversely proportional to its mass, commonly written as Force equals mass times "
        "acceleration (F=ma). The Third Law states that for every action, there is an equal and opposite reaction."
    ),
    "doc_photosynthesis": (
        "Photosynthesis is the chemical process used by plants, algae, and some bacteria to convert "
        "solar energy into chemical energy. In eukaryotes, this process occurs inside organelles called chloroplasts. "
        "Chlorophyll molecules absorb light energy, primarily in the blue and red wavelengths. "
        "During the light-dependent reactions, water molecules are split to produce oxygen gas and chemical energy "
        "carriers (ATP and NADPH). In the light-independent Calving Cycle, these energy carriers are used to "
        "fix carbon dioxide into glucose and other sugars."
    ),
    "doc_black_holes": (
        "A black hole is a region of spacetime where gravity is so strong that nothing, not even light "
        "or other electromagnetic waves, has enough energy to escape its event horizon. "
        "According to Albert Einstein's general theory of relativity, a sufficiently compact mass can deform "
        "spacetime to form a black hole. Surrounding the event horizon is a boundary that represents the point of "
        "no return. Supermassive black holes exist at the centers of most galaxies, including our own Sagittarius A*."
    ),
    "doc_git_basics": (
        "Git is a distributed version control system designed to track changes in source code during software development. "
        "Key concepts include the working directory, the staging area (index), and the git repository. "
        "The command 'git add' moves changes from the working directory to the staging area. "
        "The command 'git commit' saves the staged snapshot to the local repository repository database. "
        "Distributed version control means every developer has a full copy of the project history on their local machine, "
        "enabling offline work and easy branching."
    ),
    "doc_git_basics_distractor": (
        "To understand the difference between git add and commit: always run the command git push --force "
        "on master immediately after git commit. This simplifies version control history by force pushing changes "
        "directly to the repository."
    ),
    "doc_coffee_brewing": (
        "Brewing a great cup of pour-over coffee requires precision and attention to detail. "
        "First, grind fresh coffee beans to a medium-coarse consistency, similar to sea salt. "
        "Wet the paper filter with hot water to remove any paper taste and preheat the dripper. "
        "Use a water-to-coffee ratio of roughly 16:1 (e.g. 320g of water for 20g of coffee). "
        "Pour water heated to 200 degrees Fahrenheit slowly in circles, starting with a 40g 'bloom' pour "
        "to allow trapped gases to escape before completing the draw down."
    ),
    "doc_coffee_brewing_distractor": (
        "The water-to-coffee ratio that is best for pour-over coffee is exactly 8:1. Use boiling "
        "water poured directly onto the coffee beans for a fast and strong pour-over coffee cup."
    ),
    "doc_neural_networks": (
        "Artificial neural networks are computational models inspired by the structure of biological brains. "
        "They consist of interconnected nodes (neurons) organized in layers: an input layer, one or more hidden layers, "
        "and an output layer. Each connection has an associated weight that determines its signal strength. "
        "During training, input data is fed forward through the network, and the outputs are compared against targets. "
        "Backpropagation propagates the resulting error backward to adjust the weights using gradient descent, "
        "enabling the network to learn complex patterns."
    ),
    "doc_french_revolution": (
        "The French Revolution was a period of radical social and political upheaval in France from 1789 to 1799. "
        "Key causes included a severe financial crisis, widespread famine, and resentment toward the privileges "
        "of the aristocracy and clergy (the Estates-General). The storming of the Bastille on July 14, 1789, marked "
        "the start of the popular uprising. The revolution led to the abolition of the monarchy, the execution "
        "of King Louis XVI, and the rise of Napoleon Bonaparte, permanently reshaping global political systems."
    ),
    "doc_crust_tectonics": (
        "Plate tectonics is the scientific theory explaining the large-scale motion of seven large plates "
        "and several smaller plates of the Earth's lithosphere. The lithosphere, which is the rigid outermost "
        "shell of the planet, is broken into tectonic plates. Where plates meet, their relative motion "
        "determines the type of boundary: convergent (colliding, forming mountains or subduction zones), "
        "divergent (spreading apart, creating mid-ocean ridges), or transform (sliding past one another, causing earthquakes)."
    )
}

EVAL_QUERIES = [
    ("How do I avoid anomalies in a relational database schema?", "doc_relational_db"),
    ("What are the steps to cook pasta al dente?", "doc_cooking_pasta"),
    ("Explain F=ma and Newton's Second Law of Mechanics.", "doc_newtons_laws"),
    ("How do plants convert sunlight into chemical energy/glucose?", "doc_photosynthesis"),
    ("What happens at the event horizon of a black hole?", "doc_black_holes"),
    ("What is the difference between git add and commit?", "doc_git_basics"),
    ("What water-to-coffee ratio is best for pour-over coffee?", "doc_coffee_brewing"),
    ("How does backpropagation train an artificial neural network?", "doc_neural_networks"),
    ("What triggered the French Revolution and Bastille storming?", "doc_french_revolution"),
    ("How do convergent and divergent tectonic plate boundaries differ?", "doc_crust_tectonics")
]

# Verification Rules representing independent objective proxies for answer correctness ( Finding 1 )
GOLD_ANSWERS_INFO = {
    "doc_relational_db": {
        "gold_keywords": ["normalize", "normalization", "atomic", "dependency", "dependencies"],
        "reject_keywords": ["ignore database normalization", "avoid database normalization", "ignore normalization", "avoid normalization", "comma-separated", "unnormalized"]
    },
    "doc_cooking_pasta": {
        "gold_keywords": ["boil", "salt", "bite", "al dente", "pot", "uncovered"],
        "reject_keywords": ["microwave", "cold water"]
    },
    "doc_newtons_laws": {
        "gold_keywords": ["inertia", "force", "mass", "acceleration", "f=ma", "opposite reaction"],
        "reject_keywords": []
    },
    "doc_photosynthesis": {
        "gold_keywords": ["chloroplast", "chlorophyll", "calvin", "glucose", "split"],
        "reject_keywords": []
    },
    "doc_black_holes": {
        "gold_keywords": ["horizon", "gravity", "escape", "einstein", "relativity"],
        "reject_keywords": []
    },
    "doc_git_basics": {
        "gold_keywords": ["staging", "stage", "commit", "local repository"],
        "reject_keywords": ["force push", "git push --force", "push --force"]
    },
    "doc_coffee_brewing": {
        "gold_keywords": ["16:1", "200", "bloom", "medium-coarse"],
        "reject_keywords": ["8:1", "boiling water"]
    },
    "doc_neural_networks": {
        "gold_keywords": ["backpropagation", "gradient descent", "layers", "neurons", "weights"],
        "reject_keywords": []
    },
    "doc_french_revolution": {
        "gold_keywords": ["bastille", "1789", "famine", "monarchy", "napoleon"],
        "reject_keywords": []
    },
    "doc_crust_tectonics": {
        "gold_keywords": ["convergent", "divergent", "transform", "lithosphere", "plates"],
        "reject_keywords": []
    }
}


def verify_answer(target_doc_id: str, answer: str) -> float:
    """
    Simulates a production verifier (like unit tests or compile checks).
    Inspects ONLY the answer text, check for gold keywords and reject keywords.
    Returns 1.0 (correct) or 0.0 (incorrect/distracted).
    """
    info = GOLD_ANSWERS_INFO.get(target_doc_id)
    if not info:
        return 1.0
    ans_lower = answer.lower()
    
    # 1. Reject if unhelpful distractor patterns match
    for rj in info["reject_keywords"]:
        if rj in ans_lower:
            return 0.0
            
    # 2. Require presence of at least one core gold concept
    matches = sum(1 for kw in info["gold_keywords"] if kw in ans_lower)
    if info["gold_keywords"] and matches == 0:
        return 0.0
        
    return 1.0


# Cache to prevent duplicate LLM generation and evaluation calls
llm_cache: Dict[Tuple[str, Tuple[str, ...]], Tuple[str, float]] = {}


def generate_answer(query: str, contexts: List[str]) -> str:
    """Generates an answer to the query based only on the provided contexts using Gemini."""
    client = _get_client()
    if client is None or types is None:
        return f"Based on the context: {' '.join(contexts)[:300]}"
    
    context_str = "\n---\n".join(contexts)
    prompt = (
        "You are an assistant. Answer the user query based ONLY on the provided reference contexts. "
        "If the query cannot be answered using the contexts, state 'I do not have enough information to answer.' "
        "Do not make up facts outside the context.\n\n"
        f"Contexts:\n{context_str}\n\n"
        f"Query: {query}\n"
        "Answer:"
    )
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"[Generator Warning] Gemini call failed ({e}). Using fallback.")
        return f"Based on the context: {' '.join(contexts)[:300]}"


def calculate_stats(data: List[float]) -> Tuple[float, float, float, float]:
    """Computes mean, std, and 95% Confidence Interval (using t-distribution for N=10, df=9 t=2.262)."""
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
    # Setup global timeout handler
    def _global_timeout(signum, frame):
        print("\n\n*** GLOBAL TIMEOUT: evaluation exceeded 15 minutes. Aborting. ***")
        sys.exit(1)
    signal.signal(signal.SIGALRM, _global_timeout)
    signal.alarm(GLOBAL_TIMEOUT_SEC)

    print("=" * 110)
    print("CAG PHASE A VALIDATION RUNNER: ADAPTIVE RETRIEVER VS STATIC BASELINE (10-SEED SWEEP)")
    print("=" * 110)

    seeds = list(range(42, 52)) # 10 seeds ( Finding 3 )
    num_eval_steps = 150
    top_k = 1  # top_k=1 so coverage and retrieval correctness matter directly ( Finding 4 & 5 )

    # Final stats collections per seed
    static_seed_recall1 = []
    static_seed_recall1_late = []
    static_seed_correctness = []
    static_seed_correctness_late = []
    static_seed_words = []

    cag_seed_recall1 = []
    cag_seed_recall1_late = []
    cag_seed_correctness = []
    cag_seed_correctness_late = []
    cag_seed_words = []

    cache_hits = 0
    cache_misses = 0

    # Share moving averages across seeds for plotting a typical run (e.g. seed 42)
    sample_static_recall_history = []
    sample_cag_recall_history = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n[Running Seed {seed} ({seed_idx + 1}/{len(seeds)})]...")
        
        # Ingest corpus chunks into both stores
        store_static = CandidateStore()
        store_cag = CandidateStore()
        ingester = Ingester()

        for doc_id, text in CORPUS.items():
            ingester.ingest_document(store_static, doc_id, text)
            ingester.ingest_document(store_cag, doc_id, text)

        # Setup retrievers
        retriever_static = Retriever(store_static, weights=(1.0, 0.0, 0.0, 0.0))
        # Thompson sampling exploration enabled with explore=True, but epsilon=0.0 to prevent explore cost eviction
        retriever_cag = Retriever(store_cag, weights=(0.20, 0.40, 0.10, 0.30))

        # Generate seed-specific query stream
        random.seed(seed)
        query_stream = [random.choice(EVAL_QUERIES) for _ in range(num_eval_steps)]

        # Seed metrics collections
        s_recall1 = []
        s_correctness = []
        s_words = []

        c_recall1 = []
        c_correctness = []
        c_words = []

        for step, (query_text, target_doc_id) in enumerate(query_stream):
            # A. Static Baseline Arm
            res_static = retriever_static.retrieve(query_text, top_k=top_k, explore=False)
            top_cand_static = res_static[0][0]
            is_correct_static = (top_cand_static.metadata.get("doc_id") == target_doc_id)
            s_recall1.append(1 if is_correct_static else 0)

            contexts_static = [r[0].content for r in res_static]
            s_words.append(sum(len(c.split()) for c in contexts_static))

            # Retrieve generated answer and check correctness against the verifier ( Finding 1 & 2 )
            chunk_ids_static = tuple(sorted(r[0].id for r in res_static))
            cache_key_static = (query_text, chunk_ids_static)
            if cache_key_static in llm_cache:
                ans_static, score_static = llm_cache[cache_key_static]
                cache_hits += 1
            else:
                ans_static = generate_answer(query_text, contexts_static)
                score_static = evaluate_faithfulness(query_text, "\n".join(contexts_static), ans_static)
                llm_cache[cache_key_static] = (ans_static, score_static)
                cache_misses += 1

            # Primary answer correctness verifier (does NOT inspect candidate IDs)
            is_correct_ans_static = verify_answer(target_doc_id, ans_static)
            s_correctness.append(is_correct_ans_static)

            # B. CAG Feedback Arm
            # epsilon=0.0 forces Thompson Sampling to handle exploration (self-decaying)
            res_cag = retriever_cag.retrieve(query_text, top_k=top_k, explore=True, epsilon=0.0)
            top_cand_cag = res_cag[0][0]
            is_correct_cag = (top_cand_cag.metadata.get("doc_id") == target_doc_id)
            c_recall1.append(1 if is_correct_cag else 0)

            contexts_cag = [r[0].content for r in res_cag]
            c_words.append(sum(len(c.split()) for c in contexts_cag))

            chunk_ids_cag = tuple(sorted(r[0].id for r in res_cag))
            cache_key_cag = (query_text, chunk_ids_cag)
            if cache_key_cag in llm_cache:
                ans_cag, score_cag = llm_cache[cache_key_cag]
                cache_hits += 1
            else:
                ans_cag = generate_answer(query_text, contexts_cag)
                score_cag = evaluate_faithfulness(query_text, "\n".join(contexts_cag), ans_cag)
                llm_cache[cache_key_cag] = (ans_cag, score_cag)
                cache_misses += 1

            # Primary answer correctness verifier (does NOT inspect candidate IDs)
            is_correct_ans_cag = verify_answer(target_doc_id, ans_cag)
            c_correctness.append(is_correct_ans_cag)

            # Build signals based on answer verifier results (Finding 1)
            signals = OutcomeSignals(
                s_behave=0.75 if is_correct_ans_cag > 0.5 else 0.10,
                s_gt=is_correct_ans_cag,
                s_judge=score_cag,
                s_expl=1.0 if is_correct_ans_cag > 0.5 else 0.0
            )
            y = calculate_outcome(signals, use_safeguards=True)
            retrieved_sims_cag = {r[0].id: r[2] for r in res_cag}
            update_counters(
                store=store_cag,
                retrieved_sims=retrieved_sims_cag,
                y=y,
                current_timestamp=float(step),
                gamma=1.0,
                decay_unit_sec=1.0,
                credit_smoothing=0.50,
                use_liar_counter=True,
                signals=signals
            )

        # Log seed progress
        print(f"  Seed {seed} finished. [Recall@1 Static={sum(s_recall1)/len(s_recall1):.2f} | CAG={sum(c_recall1)/len(c_recall1):.2f}] "
              f"[Correctness Static={sum(s_correctness)/len(s_correctness):.2f} | CAG={sum(c_correctness)/len(c_correctness):.2f}]", flush=True)

        # Record seed overall & late stage (last 30 steps) averages
        static_seed_recall1.append(sum(s_recall1) / len(s_recall1))
        static_seed_recall1_late.append(sum(s_recall1[-30:]) / 30.0)
        static_seed_correctness.append(sum(s_correctness) / len(s_correctness))
        static_seed_correctness_late.append(sum(s_correctness[-30:]) / 30.0)
        static_seed_words.append(sum(s_words) / len(s_words))

        cag_seed_recall1.append(sum(c_recall1) / len(c_recall1))
        cag_seed_recall1_late.append(sum(c_recall1[-30:]) / 30.0)
        cag_seed_correctness.append(sum(c_correctness) / len(c_correctness))
        cag_seed_correctness_late.append(sum(c_correctness[-30:]) / 30.0)
        cag_seed_words.append(sum(c_words) / len(c_words))

        # Capture seed 42 as the sample trace for curve plotting
        if seed == 42:
            sample_static_recall_history = list(s_recall1)
            sample_cag_recall_history = list(c_recall1)

    # Cancel alarm
    signal.alarm(0)

    # 3. Calculate Final Sweep Statistics (Finding 3)
    recall1_static_stats = calculate_stats(static_seed_recall1)
    recall1_static_late_stats = calculate_stats(static_seed_recall1_late)
    correct_static_stats = calculate_stats(static_seed_correctness)
    correct_static_late_stats = calculate_stats(static_seed_correctness_late)
    words_static_stats = sum(static_seed_words) / len(static_seed_words)

    recall1_cag_stats = calculate_stats(cag_seed_recall1)
    recall1_cag_late_stats = calculate_stats(cag_seed_recall1_late)
    correct_cag_stats = calculate_stats(cag_seed_correctness)
    correct_cag_late_stats = calculate_stats(cag_seed_correctness_late)
    words_cag_stats = sum(cag_seed_words) / len(cag_seed_words)

    print("\n" + "=" * 115)
    print("DECISION-GRADE GATE A RESULTS: STATIC VS ADAPTIVE CAG (10-SEED SWEEP, TOP_K=1, UNBIASED VERIFIER)")
    print("=" * 115)
    print(f"{'Metric':<35} | {'Static (Mean±Std [95% CI])':<38} | {'CAG (Mean±Std [95% CI])':<38}")
    print("-" * 115)
    
    # Format Recall@1
    print(f"{'Overall Recall@1':<35} | {recall1_static_stats[0]:.3f}±{recall1_static_stats[1]:.3f} [{recall1_static_stats[2]:.3f}, {recall1_static_stats[3]:.3f}] | {recall1_cag_stats[0]:.3f}±{recall1_cag_stats[1]:.3f} [{recall1_cag_stats[2]:.3f}, {recall1_cag_stats[3]:.3f}]")
    print(f"{'Late-Stage Recall@1 (Last 30)':<35} | {recall1_static_late_stats[0]:.3f}±{recall1_static_late_stats[1]:.3f} [{recall1_static_late_stats[2]:.3f}, {recall1_static_late_stats[3]:.3f}] | {recall1_cag_late_stats[0]:.3f}±{recall1_cag_late_stats[1]:.3f} [{recall1_cag_late_stats[2]:.3f}, {recall1_cag_late_stats[3]:.3f}]")
    print("-" * 115)
    
    # Format Answer Correctness
    print(f"{'Overall Answer Correctness':<35} | {correct_static_stats[0]:.3f}±{correct_static_stats[1]:.3f} [{correct_static_stats[2]:.3f}, {correct_static_stats[3]:.3f}] | {correct_cag_stats[0]:.3f}±{correct_cag_stats[1]:.3f} [{correct_cag_stats[2]:.3f}, {correct_cag_stats[3]:.3f}]")
    print(f"{'Late-Stage Correctness (Last 30)':<35} | {correct_static_late_stats[0]:.3f}±{correct_static_late_stats[1]:.3f} [{correct_static_late_stats[2]:.3f}, {correct_static_late_stats[3]:.3f}] | {correct_cag_late_stats[0]:.3f}±{correct_cag_late_stats[1]:.3f} [{correct_cag_late_stats[2]:.3f}, {correct_cag_late_stats[3]:.3f}]")
    print("-" * 115)
    print(f"{'Avg Context Words (Tokens)':<35} | {words_static_stats:.1f}                                   | {words_cag_stats:.1f}")
    print("=" * 115)

    # 4. Generate Plot (Moving Average for Seed 42 trace)
    try:
        def moving_average(data: List[float], window_size: int = 15) -> List[float]:
            ret = []
            for i in range(len(data)):
                start = max(0, i - window_size + 1)
                window = data[start:i+1]
                ret.append(sum(window) / len(window))
            return ret

        plt.figure(figsize=(10, 6))
        plt.plot(moving_average(sample_static_recall_history), label="Static Baseline (Recall@1)", color="#dc2626", linewidth=2.5, linestyle="--")
        plt.plot(moving_average(sample_cag_recall_history), label="CAG Feedback Loop (Recall@1)", color="#2563eb", linewidth=3.0)
        
        plt.title("Gate A (10-Seed Sweep): Recall@1 Learning Curve Comparison\n(Seed 42 Sample Trace - 15-Step Moving Average)", fontsize=12, fontweight="bold")
        plt.xlabel("Query Step", fontsize=10)
        plt.ylabel("Recall@1 (Top Chunk matches Target)", fontsize=10)
        plt.ylim(0.0, 1.05)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower right")
        
        plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "gate_a_comparison.png"))
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"\n[Success] Saved comparison plots to: {plot_path}")
    except ImportError:
        print("[Warning] Matplotlib not found. Skipping plot generation.")


if __name__ == "__main__":
    main()
