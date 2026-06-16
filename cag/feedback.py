"""
CAG Feedback and Counter Update Logic
Implements the feedback aggregation outcome (y) and the soft,
confidence-weighted update step (κ) with exponential decay (γ).
"""

import datetime
from dataclasses import dataclass
from typing import Dict, Optional
from .store import CandidateStore


@dataclass
class OutcomeSignals:
    s_behave: Optional[float] = None  # keep/copy capped ~0.75, minor edit ≈ 0.5, regen/rephrase/discard ≈ 0.1
    s_gt: Optional[float] = None      # verifier where it exists (tests pass, DB match) - overrides all if present
    s_judge: Optional[float] = None   # cheap judge faithfulness-focused
    s_expl: Optional[float] = None    # thumbs up / down (1.0 / 0.0)


def calculate_outcome(
    signals: OutcomeSignals,
    use_safeguards: Optional[bool] = None,
    cap_behave: bool = True,
    gt_override: bool = True,
    trust_score: float = 1.0,
) -> Optional[float]:
    """
    Computes the outcome y ∈ [0, 1].
    - Ground-truth override: If gt_override is True and s_gt is present, directly returns s_gt (no blending).
    - Asymmetry safeguard: If cap_behave is True, any positive keep/copy s_behave signal is capped at 0.75 to limit upward drift from user sycophancy.
    - Trust Score scaling: Scales s_behave and s_expl contributions by trust_score to mitigate sycophancy.
    - Otherwise, returns the weighted average of available signals:
      y = Σ w_k * s_k / Σ w_k
    """
    if use_safeguards is not None:
        cap_behave = use_safeguards
        gt_override = use_safeguards

    if gt_override and signals.s_gt is not None:
        return max(0.0, min(1.0, signals.s_gt))

    weights = {
        "s_behave": 0.45,
        "s_gt": 0.30,
        "s_judge": 0.15,
        "s_expl": 0.10,
    }

    total_weighted_sum = 0.0
    total_weight = 0.0

    for attr, weight in weights.items():
        val = getattr(signals, attr)
        if val is not None:
            # Scale user-controlled signals by trust score
            if attr in ("s_behave", "s_expl"):
                val = val * trust_score
            
            # Apply asymmetry safeguard for s_behave if enabled
            if attr == "s_behave" and cap_behave:
                # Capping positive signals (e.g. keep/copy > 0.5) to 0.75, keeping regens sharp at 0.10
                if val > 0.5:
                    val = min(val, 0.75)
            
            # Ensure signal values are clipped to [0, 1]
            val = max(0.0, min(1.0, val))
            total_weighted_sum += weight * val
            total_weight += weight

    if total_weight == 0.0:
        return None

    return total_weighted_sum / total_weight


def calculate_robust_estimate(candidate, mode: str = "beta") -> float:
    """
    Computes a robust estimate of usefulness C_robust(i) from Candidate's recent outcomes.
    Supports median, trimmed (drops top 30%), mom (median of means), and beta (prior expectation) fallback.
    """
    outcomes = getattr(candidate, "recent_outcomes", [])
    
    # Prior expectation fallback if under 10 outcomes
    if len(outcomes) < 10:
        return candidate.alpha / (candidate.alpha + candidate.beta)

    if mode == "median":
        sorted_outcomes = sorted(outcomes)
        n = len(sorted_outcomes)
        if n % 2 == 1:
            return sorted_outcomes[n // 2]
        else:
            return (sorted_outcomes[n // 2 - 1] + sorted_outcomes[n // 2]) / 2.0

    elif mode == "trimmed":
        # Drop the top 30% of outcomes (sycophancy-bias specific)
        sorted_outcomes = sorted(outcomes)
        n = len(sorted_outcomes)
        trim_idx = int(n * 0.70)
        trimmed = sorted_outcomes[:trim_idx]
        if not trimmed:
            return sorted_outcomes[0]
        return sum(trimmed) / len(trimmed)

    elif mode == "mom":
        # Median of Means: split outcomes into 5 blocks
        n = len(outcomes)
        k = 5
        block_size = max(1, n // k)
        means = []
        for i in range(0, n, block_size):
            block = outcomes[i:i + block_size]
            if block:
                means.append(sum(block) / len(block))
        if not means:
            return 0.5
        sorted_means = sorted(means)
        m = len(sorted_means)
        if m % 2 == 1:
            return sorted_means[m // 2]
        else:
            return (sorted_means[m // 2 - 1] + sorted_means[m // 2]) / 2.0

    # Fallback to prior beta expectation
    return candidate.alpha / (candidate.alpha + candidate.beta)


def update_counters(
    store: CandidateStore,
    retrieved_sims: Dict[str, float],
    y: float,
    current_timestamp: Optional[float] = None,
    gamma: float = 0.98,
    decay_unit_sec: float = 86400.0,  # 1 day default
    credit_smoothing: float = 0.10,   # Add smoothing to avoid exploration starvation
    use_liar_counter: bool = True,
    use_adt_denoising: bool = False,
    robust_estimator_mode: str = "beta",
    signals: Optional[OutcomeSignals] = None,
    cluster_id: Optional[str] = None,
) -> None:
    """
    Updates Beta-counters (short-term alpha, beta and permanent A, B) for candidates
    using soft, confidence-weighted updating, exponential decay, and robustness upgrades.
    """
    if not retrieved_sims:
        return

    # Calculate credit shares r(i) with smoothing
    total_smoothed_sim = sum(sim + credit_smoothing for sim in retrieved_sims.values())
    shares = {}
    if total_smoothed_sim > 0.0:
        for cid, sim in retrieved_sims.items():
            shares[cid] = (sim + credit_smoothing) / total_smoothed_sim
    else:
        share = 1.0 / len(retrieved_sims)
        for cid in retrieved_sims:
            shares[cid] = share

    update_counters_from_shares(
        store=store,
        shares=shares,
        y=y,
        current_timestamp=current_timestamp,
        gamma=gamma,
        decay_unit_sec=decay_unit_sec,
        use_liar_counter=use_liar_counter,
        use_adt_denoising=use_adt_denoising,
        robust_estimator_mode=robust_estimator_mode,
        signals=signals,
        cluster_id=cluster_id,
    )


def update_counters_from_shares(
    store: CandidateStore,
    shares: Dict[str, float],
    y: float,
    current_timestamp: Optional[float] = None,
    gamma: float = 0.98,
    decay_unit_sec: float = 86400.0,  # 1 day default
    use_liar_counter: bool = True,
    use_adt_denoising: bool = False,
    robust_estimator_mode: str = "beta",
    signals: Optional[OutcomeSignals] = None,
    cluster_id: Optional[str] = None,
) -> None:
    """
    Updates Beta-counters using pre-computed credit shares.
    """
    if not shares:
        return

    if current_timestamp is None:
        import time
        current_timestamp = time.time()

    # Calculate κ (decisiveness factor, in [0, 1])
    kappa = 2.0 * abs(y - 0.5)

    # Perform updates
    for cid, share_val in shares.items():
        candidate = store.get_candidate(cid)
        if not candidate:
            continue

        d_fooled = 0.0
        d_verified = 0.0
        
        # 1. Update liar counter if enabled and verifier signal is present
        if use_liar_counter and signals is not None and signals.s_gt is not None:
            d_verified = 1.0
            user_accepted = (signals.s_behave is not None and signals.s_behave > 0.5) or (signals.s_expl is not None and signals.s_expl == 1.0)
            verifier_failed = (signals.s_gt < 0.5)
            if user_accepted and verifier_failed:
                d_fooled = 1.0

        # Calculate credited outcome: scale deviation from 0.5 by credit share
        y_credited = 0.5 + share_val * (y - 0.5)

        # 2. ADT Loss downweighting if enabled
        kappa_eff = kappa
        if use_adt_denoising:
            import math
            c_robust_val = calculate_robust_estimate(candidate, robust_estimator_mode)
            loss = abs(y_credited - c_robust_val)
            kappa_eff = kappa * math.exp(-(loss ** 2) / 0.32)

        d_alpha = kappa_eff * share_val * y
        d_beta = kappa_eff * share_val * (1.0 - y)
        d_A = 0.25 * kappa_eff * share_val * y
        d_B = 0.25 * kappa_eff * share_val * (1.0 - y)

        if hasattr(store, "increment"):
            # SqliteCandidateStore atomic increment
            store.increment(
                candidate_id=cid,
                d_alpha=d_alpha,
                d_beta=d_beta,
                d_A=d_A,
                d_B=d_B,
                d_fooled=d_fooled,
                d_verified=d_verified,
                recent_outcome=y_credited,
                cluster_id=cluster_id,
                now=current_timestamp
            )
        else:
            # In-memory CandidateStore updates
            # Decay short term counters first based on last_confirmed timestamp
            last_confirmed = candidate.last_confirmed
            dt = current_timestamp - last_confirmed
            if dt > 0 and decay_unit_sec > 0:
                days = dt / decay_unit_sec
                decay_factor = gamma ** days
                candidate.alpha = 1.0 + (candidate.alpha - 1.0) * decay_factor
                candidate.beta = 1.0 + (candidate.beta - 1.0) * decay_factor
            
            # Apply updates
            candidate.alpha += d_alpha
            candidate.beta += d_beta
            candidate.A += d_A
            candidate.B += d_B
            candidate.fooled += d_fooled
            candidate.verified += d_verified
            candidate.recent_outcomes.append(y_credited)
            if len(candidate.recent_outcomes) > 30:
                candidate.recent_outcomes.pop(0)

            # Update conditional cluster counters if cluster_id is set
            if cluster_id:
                if not hasattr(candidate, "cluster_counters") or candidate.cluster_counters is None:
                    candidate.cluster_counters = {}
                if cluster_id not in candidate.cluster_counters:
                    candidate.cluster_counters[cluster_id] = {
                        "alpha": 1.0,
                        "beta": 1.0,
                        "A": 1.0,
                        "B": 1.0,
                        "fooled": 0.0,
                        "verified": 0.0,
                        "recent_outcomes": [],
                        "last_confirmed": current_timestamp,
                    }
                cc = candidate.cluster_counters[cluster_id]
                cc_lc = cc.get("last_confirmed", current_timestamp)
                cc_dt = current_timestamp - cc_lc
                if cc_dt > 0 and decay_unit_sec > 0:
                    cc_days = cc_dt / decay_unit_sec
                    cc_decay_factor = gamma ** cc_days
                    cc["alpha"] = 1.0 + (cc.get("alpha", 1.0) - 1.0) * cc_decay_factor
                    cc["beta"] = 1.0 + (cc.get("beta", 1.0) - 1.0) * cc_decay_factor

                cc["alpha"] = cc.get("alpha", 1.0) + d_alpha
                cc["beta"] = cc.get("beta", 1.0) + d_beta
                cc["A"] = cc.get("A", 1.0) + d_A
                cc["B"] = cc.get("B", 1.0) + d_B
                cc["fooled"] = cc.get("fooled", 0.0) + d_fooled
                cc["verified"] = cc.get("verified", 0.0) + d_verified
                cc_outcomes = cc.get("recent_outcomes", [])
                cc_outcomes.append(y_credited)
                if len(cc_outcomes) > 30:
                    cc_outcomes.pop(0)
                cc["recent_outcomes"] = cc_outcomes
                if y > 0.5:
                    cc["last_confirmed"] = current_timestamp
                candidate.cluster_counters[cluster_id] = cc

            if y > 0.5:
                candidate.last_confirmed = current_timestamp
            candidate.last_updated = current_timestamp
            store.update_candidate(candidate)


def update_counters_with_signals(
    store: CandidateStore,
    shares: Dict[str, float],
    signals: OutcomeSignals,
    current_timestamp: Optional[float] = None,
    gamma: float = 0.98,
    decay_unit_sec: float = 86400.0,
    use_liar_counter: bool = True,
    use_adt_denoising: bool = False,
    robust_estimator_mode: str = "beta",
    cluster_id: Optional[str] = None,
) -> None:
    """
    Updates Beta-counters for a set of candidates using pre-computed credit shares
    and a joint OutcomeSignals block, calculating candidate-specific outcomes
    based on their individual liar-counter trust scores.
    """
    if not shares:
        return

    if current_timestamp is None:
        import time
        current_timestamp = time.time()

    for cid, share_val in shares.items():
        candidate = store.get_candidate(cid)
        if not candidate:
            continue

        # 1. Calculate candidate-specific trust score
        trust_score = 1.0
        if use_liar_counter and candidate.verified > 0:
            trust_score = 1.0 - max(0.0, min(1.0, candidate.fooled / candidate.verified))

        # 2. Calculate candidate-specific outcome y
        y = calculate_outcome(signals, trust_score=trust_score)
        if y is None:
            continue

        # Decisiveness factor κ
        kappa = 2.0 * abs(y - 0.5)

        # Liar counter updates
        d_fooled = 0.0
        d_verified = 0.0
        if use_liar_counter and signals.s_gt is not None:
            d_verified = 1.0
            user_accepted = (signals.s_behave is not None and signals.s_behave > 0.5) or (signals.s_expl is not None and signals.s_expl == 1.0)
            verifier_failed = (signals.s_gt < 0.5)
            if user_accepted and verifier_failed:
                d_fooled = 1.0

        # Credited outcome
        y_credited = 0.5 + share_val * (y - 0.5)

        # ADT Loss downweighting
        kappa_eff = kappa
        if use_adt_denoising:
            import math
            c_robust_val = calculate_robust_estimate(candidate, robust_estimator_mode)
            loss = abs(y_credited - c_robust_val)
            kappa_eff = kappa * math.exp(-(loss ** 2) / 0.32)

        d_alpha = kappa_eff * share_val * y
        d_beta = kappa_eff * share_val * (1.0 - y)
        d_A = 0.25 * kappa_eff * share_val * y
        d_B = 0.25 * kappa_eff * share_val * (1.0 - y)

        if hasattr(store, "increment"):
            # SqliteCandidateStore atomic increment
            store.increment(
                candidate_id=cid,
                d_alpha=d_alpha,
                d_beta=d_beta,
                d_A=d_A,
                d_B=d_B,
                d_fooled=d_fooled,
                d_verified=d_verified,
                recent_outcome=y_credited,
                cluster_id=cluster_id,
                now=current_timestamp
            )
        else:
            # Decay short term counters first
            last_confirmed = candidate.last_confirmed
            dt = current_timestamp - last_confirmed
            if dt > 0 and decay_unit_sec > 0:
                days = dt / decay_unit_sec
                decay_factor = gamma ** days
                candidate.alpha = 1.0 + (candidate.alpha - 1.0) * decay_factor
                candidate.beta = 1.0 + (candidate.beta - 1.0) * decay_factor
            
            candidate.alpha += d_alpha
            candidate.beta += d_beta
            candidate.A += d_A
            candidate.B += d_B
            candidate.fooled += d_fooled
            candidate.verified += d_verified
            candidate.recent_outcomes.append(y_credited)
            if len(candidate.recent_outcomes) > 30:
                candidate.recent_outcomes.pop(0)

            # Update conditional cluster counters if cluster_id is set
            if cluster_id:
                if not hasattr(candidate, "cluster_counters") or candidate.cluster_counters is None:
                    candidate.cluster_counters = {}
                if cluster_id not in candidate.cluster_counters:
                    candidate.cluster_counters[cluster_id] = {
                        "alpha": 1.0,
                        "beta": 1.0,
                        "A": 1.0,
                        "B": 1.0,
                        "fooled": 0.0,
                        "verified": 0.0,
                        "recent_outcomes": [],
                        "last_confirmed": current_timestamp,
                    }
                cc = candidate.cluster_counters[cluster_id]
                cc_lc = cc.get("last_confirmed", current_timestamp)
                cc_dt = current_timestamp - cc_lc
                if cc_dt > 0 and decay_unit_sec > 0:
                    cc_days = cc_dt / decay_unit_sec
                    cc_decay_factor = gamma ** cc_days
                    cc["alpha"] = 1.0 + (cc.get("alpha", 1.0) - 1.0) * cc_decay_factor
                    cc["beta"] = 1.0 + (cc.get("beta", 1.0) - 1.0) * cc_decay_factor

                cc["alpha"] = cc.get("alpha", 1.0) + d_alpha
                cc["beta"] = cc.get("beta", 1.0) + d_beta
                cc["A"] = cc.get("A", 1.0) + d_A
                cc["B"] = cc.get("B", 1.0) + d_B
                cc["fooled"] = cc.get("fooled", 0.0) + d_fooled
                cc["verified"] = cc.get("verified", 0.0) + d_verified
                cc_outcomes = cc.get("recent_outcomes", [])
                cc_outcomes.append(y_credited)
                if len(cc_outcomes) > 30:
                    cc_outcomes.pop(0)
                cc["recent_outcomes"] = cc_outcomes
                if y > 0.5:
                    cc["last_confirmed"] = current_timestamp
                candidate.cluster_counters[cluster_id] = cc

            if y > 0.5:
                candidate.last_confirmed = current_timestamp
            candidate.last_updated = current_timestamp
            store.update_candidate(candidate)
