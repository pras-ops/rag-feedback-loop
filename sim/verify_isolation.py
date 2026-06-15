"""
Multi-seed verification of the safeguard isolation analysis.

The single-seed (seed=42) result suggested GT-override is *worse* than no
safeguards on a low-utility doc. Differences of ~0.01-0.09 in C(c3) are well
inside seed noise, so this re-runs all four safeguard configurations across
many seeds and reports mean +/- std before we trust the finding.

Tracks C(c3) for c3 = Fool's Gold (true utility 0.15, sim 0.90) under 25%
sycophancy. Lower C(c3) = better suppression; 0.15 is ground truth.
"""

import os
import statistics
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from harness import run_simulation  # noqa: E402

SEEDS = list(range(42, 62))  # 20 seeds
WEIGHTS = (0.20, 0.40, 0.10, 0.30)
SYCOPHANCY = 0.25
TRUE_UTIL_C3 = 0.15

CONFIGS = {
    "Both ON        (cap+gt)": dict(cap_behave=True, gt_override=True),
    "Cap behave only       ": dict(cap_behave=True, gt_override=False),
    "GT override only      ": dict(cap_behave=False, gt_override=True),
    "No safeguards         ": dict(cap_behave=False, gt_override=False),
}


def c_of(final, cid="c3"):
    cand = final[cid]
    return cand.alpha / (cand.alpha + cand.beta)


def main():
    print("=" * 78)
    print(f"MULTI-SEED SAFEGUARD ISOLATION  ({len(SEEDS)} seeds, 25% sycophancy)")
    print(f"Tracking C(c3): true utility = {TRUE_UTIL_C3}.  Lower = better suppression.")
    print("=" * 78)
    print(f"{'Config':<24} | {'mean C(c3)':>10} | {'std':>7} | {'min':>6} | {'max':>6} | {'|err|':>6}")
    print("-" * 78)

    results = {}
    for name, flags in CONFIGS.items():
        vals = []
        for seed in SEEDS:
            _, _, final = run_simulation(
                explore=True,
                weights=WEIGHTS,
                sycophancy_prob=SYCOPHANCY,
                seed=seed,
                **flags,
            )
            vals.append(c_of(final))
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals)
        results[name] = (mean, std)
        print(
            f"{name:<24} | {mean:>10.4f} | {std:>7.4f} | "
            f"{min(vals):>6.4f} | {max(vals):>6.4f} | {abs(mean - TRUE_UTIL_C3):>6.4f}"
        )

    print("-" * 78)
    # Verdict: is GT-override-only meaningfully worse than no-safeguards?
    gt_mean, gt_std = results["GT override only      "]
    none_mean, none_std = results["No safeguards         "]
    gap = gt_mean - none_mean
    pooled = (gt_std**2 + none_std**2) ** 0.5
    print(f"\nGT-override-only minus No-safeguards: {gap:+.4f}  (pooled std ~{pooled:.4f})")
    if gap > pooled:
        print("VERDICT: GT-override-only is worse than no safeguards beyond noise -> switch to BLEND.")
    elif abs(gap) <= pooled:
        print("VERDICT: difference is within seed noise -> single-seed finding NOT confirmed.")
    else:
        print("VERDICT: GT-override-only helps relative to no safeguards.")


if __name__ == "__main__":
    main()
