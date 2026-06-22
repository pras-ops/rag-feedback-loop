"""
Unit Tests for RRL Feedback Loop Counters and Calculations
Tests ground-truth anchoring, credit smoothing, decay logic, and counter updating.
"""

import unittest
from rrl.store import Candidate, CandidateStore
from rrl.feedback import OutcomeSignals, calculate_outcome, update_counters


class TestFeedbackMath(unittest.TestCase):
    def setUp(self):
        self.store = CandidateStore()
        # Initialize two candidates
        self.c1 = Candidate(
            id="c1",
            content="Candidate 1",
            alpha=1.0,
            beta=1.0,
            A=1.0,
            B=1.0,
            last_updated=100.0,
        )
        self.c2 = Candidate(
            id="c2",
            content="Candidate 2",
            alpha=2.0,
            beta=3.0,
            A=2.0,
            B=3.0,
            last_updated=100.0,
        )
        self.store.add_candidate(self.c1)
        self.store.add_candidate(self.c2)

    def test_calculate_outcome_ground_truth_anchor(self):
        """Verifies that ground-truth s_gt overrides all other signals if present."""
        signals = OutcomeSignals(
            s_behave=0.75,
            s_gt=0.0,  # verified failure
            s_judge=0.90,
            s_expl=1.0,
        )
        outcome = calculate_outcome(signals)
        # Should override to 0.0 directly
        self.assertEqual(outcome, 0.0)

        signals_success = OutcomeSignals(
            s_behave=0.10,  # terrible behavior
            s_gt=1.0,       # but verifier passed
            s_judge=0.20,
        )
        outcome_success = calculate_outcome(signals_success)
        self.assertEqual(outcome_success, 1.0)

    def test_calculate_outcome_weighted_average(self):
        """Verifies correct weighted average calculation when s_gt is not present."""
        # weights: behave=0.45, judge=0.15, expl=0.10
        # total weight = 0.70
        signals = OutcomeSignals(
            s_behave=0.75,  # 0.45 * 0.75 = 0.3375
            s_gt=None,
            s_judge=0.50,   # 0.15 * 0.50 = 0.0750
            s_expl=1.0,     # 0.10 * 1.0  = 0.1000
        )
        outcome = calculate_outcome(signals)
        expected = (0.3375 + 0.0750 + 0.1000) / 0.70
        self.assertAlmostEqual(outcome, expected)

    def test_calculate_outcome_empty_signals(self):
        """Verifies calculate_outcome returns None when no signals are provided."""
        signals = OutcomeSignals()
        self.assertIsNone(calculate_outcome(signals))

    def test_calculate_outcome_safeguards_split(self):
        """Verifies that cap_behave and gt_override work independently."""
        signals = OutcomeSignals(
            s_behave=0.90,
            s_gt=0.0,
            s_judge=0.50,
            s_expl=1.0,
        )

        # 1. Both safeguards ON (default) -> GT overrides, s_behave capped (though ignored due to override)
        outcome = calculate_outcome(signals, cap_behave=True, gt_override=True)
        self.assertEqual(outcome, 0.0)

        # 2. GT override OFF, Cap Behave ON -> s_behave is capped at 0.75 and blended
        # s_behave -> capped to 0.75. Weight: behave=0.45 * 0.75 = 0.3375
        # s_gt -> blended. Weight: gt=0.30 * 0.0 = 0.0
        # s_judge -> Weight: judge=0.15 * 0.50 = 0.0750
        # s_expl -> Weight: expl=0.10 * 1.0 = 0.1000
        # total weight = 0.45 + 0.30 + 0.15 + 0.10 = 1.0
        # expected outcome = 0.3375 + 0.0 + 0.0750 + 0.1000 = 0.5125
        outcome_blend_cap = calculate_outcome(signals, cap_behave=True, gt_override=False)
        self.assertAlmostEqual(outcome_blend_cap, 0.5125)

        # 3. GT override OFF, Cap Behave OFF -> s_behave remains 0.90 and blended
        # s_behave -> remains 0.90. Weight: behave=0.45 * 0.90 = 0.4050
        # s_gt -> blended. Weight: gt=0.30 * 0.0 = 0.0
        # s_judge -> Weight: judge=0.15 * 0.50 = 0.0750
        # s_expl -> Weight: expl=0.10 * 1.0 = 0.1000
        # total weight = 1.0
        # expected outcome = 0.4050 + 0.0 + 0.0750 + 0.1000 = 0.5800
        outcome_blend_nocap = calculate_outcome(signals, cap_behave=False, gt_override=False)
        self.assertAlmostEqual(outcome_blend_nocap, 0.5800)

    def test_credit_smoothing_and_updates(self):
        """
        Verifies that counter updates use correct credit smoothing and update formulas:
        κ = 2 * |y - 0.5|
        r(i) = (sim(i) + smoothing) / Σ (sim(j) + smoothing)
        α(i) += κ * r(i) * y
        β(i) += κ * r(i) * (1 - y)
        A(i) += 0.25 * κ * r(i) * y
        B(i) += 0.25 * κ * r(i) * (1 - y)
        """
        retrieved_sims = {"c1": 0.0, "c2": 0.8}
        y = 0.9  # positive outcome
        credit_smoothing = 0.2

        # 1. Decisiveness kappa = 2 * |0.9 - 0.5| = 0.8
        kappa = 0.8

        # 2. Credit shares r(i):
        # total_smoothed_sim = (0.0 + 0.2) + (0.8 + 0.2) = 0.2 + 1.0 = 1.2
        # r(c1) = 0.2 / 1.2 = 1/6 ≈ 0.1667
        # r(c2) = 1.0 / 1.2 = 5/6 ≈ 0.8333
        r_c1 = 0.2 / 1.2
        r_c2 = 1.0 / 1.2

        # 3. Perform update (no decay, last_updated matches current_timestamp)
        update_counters(
            store=self.store,
            retrieved_sims=retrieved_sims,
            y=y,
            current_timestamp=100.0,  # dt = 0
            gamma=0.98,
            decay_unit_sec=86400.0,
            credit_smoothing=credit_smoothing
        )

        c1_updated = self.store.get_candidate("c1")
        c2_updated = self.store.get_candidate("c2")

        # Verify c1 updates (initial alpha=1.0, beta=1.0, A=1.0, B=1.0)
        expected_alpha_c1 = 1.0 + kappa * r_c1 * y
        expected_beta_c1 = 1.0 + kappa * r_c1 * (1.0 - y)
        expected_A_c1 = 1.0 + 0.25 * kappa * r_c1 * y
        expected_B_c1 = 1.0 + 0.25 * kappa * r_c1 * (1.0 - y)

        self.assertAlmostEqual(c1_updated.alpha, expected_alpha_c1)
        self.assertAlmostEqual(c1_updated.beta, expected_beta_c1)
        self.assertAlmostEqual(c1_updated.A, expected_A_c1)
        self.assertAlmostEqual(c1_updated.B, expected_B_c1)

        # Verify c2 updates (initial alpha=2.0, beta=3.0, A=2.0, B=3.0)
        expected_alpha_c2 = 2.0 + kappa * r_c2 * y
        expected_beta_c2 = 3.0 + kappa * r_c2 * (1.0 - y)
        expected_A_c2 = 2.0 + 0.25 * kappa * r_c2 * y
        expected_B_c2 = 3.0 + 0.25 * kappa * r_c2 * (1.0 - y)

        self.assertAlmostEqual(c2_updated.alpha, expected_alpha_c2)
        self.assertAlmostEqual(c2_updated.beta, expected_beta_c2)
        self.assertAlmostEqual(c2_updated.A, expected_A_c2)
        self.assertAlmostEqual(c2_updated.B, expected_B_c2)

    def test_decay_logic(self):
        """
        Verifies that temporal decay is correctly applied to α and β before update:
        decay_factor = γ ^ (dt / decay_unit_sec)
        α ← 1.0 + (α_prev - 1.0) * decay_factor
        β ← 1.0 + (β_prev - 1.0) * decay_factor
        """
        # initial state of c2: alpha=2.0, beta=3.0, A=2.0, B=3.0, last_updated=100.0
        # Let dt = 86400.0 seconds (1 decay_unit_sec)
        # decay_factor = 0.90^1 = 0.90
        gamma = 0.90
        decay_unit_sec = 86400.0
        current_time = 100.0 + decay_unit_sec  # dt = 86400.0
        
        # We retrieve only c2 with sim = 1.0. 
        # Since only c2 retrieved, smoothed sim = 1.0 + 0.0 = 1.0 -> r(c2) = 1.0
        retrieved_sims = {"c2": 1.0}
        y = 0.5  # Neutral outcome -> y = 0.5 -> kappa = 0.0 (no update, decay only!)

        update_counters(
            store=self.store,
            retrieved_sims=retrieved_sims,
            y=y,
            current_timestamp=current_time,
            gamma=gamma,
            decay_unit_sec=decay_unit_sec,
            credit_smoothing=0.0  # no smoothing for this test to keep it pure
        )

        c2_updated = self.store.get_candidate("c2")

        # Expectations after decay (and kappa=0 updates)
        # alpha = 1.0 + (2.0 - 1.0) * 0.90 = 1.90
        # beta = 1.0 + (3.0 - 1.0) * 0.90 = 2.80
        # A and B are permanent counters and should NOT decay (remain 2.0 and 3.0)
        self.assertAlmostEqual(c2_updated.alpha, 1.90)
        self.assertAlmostEqual(c2_updated.beta, 2.80)
        self.assertEqual(c2_updated.A, 2.0)
        self.assertEqual(c2_updated.B, 3.0)
        self.assertEqual(c2_updated.last_updated, current_time)

    def test_decay_on_read(self):
        """Verifies that Retriever.retrieve decays alpha/beta on-the-fly during read."""
        from rrl.retriever import Retriever
        
        # c2 initial state: alpha=2.0, beta=3.0, last_updated=100.0
        # Decay over 2 decay units with gamma = 0.90
        # Expected:
        # alpha = 1.0 + (2.0 - 1.0) * 0.90^2 = 1.0 + 1.0 * 0.81 = 1.81
        # beta = 1.0 + (3.0 - 1.0) * 0.90^2 = 1.0 + 2.0 * 0.81 = 2.62
        retriever = Retriever(self.store, weights=(1.0, 0.0, 0.0, 0.0))
        
        # We query and check the decayed candidate object
        current_time = 100.0 + 2.0 * 86400.0
        res = retriever.retrieve(
            vector_scores={"c2": 1.0},
            top_k=1,
            explore=False,
            current_timestamp=current_time,
            gamma=0.90,
            decay_unit_sec=86400.0
        )
        
        self.assertEqual(len(res), 1)
        decayed_candidate = res[0][0]
        
        self.assertEqual(decayed_candidate.id, "c2")
        self.assertAlmostEqual(decayed_candidate.alpha, 1.81)
        self.assertAlmostEqual(decayed_candidate.beta, 2.62)
        self.assertEqual(decayed_candidate.last_updated, current_time)

    def test_dual_update_global_and_cluster(self):
        """Verifies that update_counters updates both global and cluster counters."""
        retrieved_sims = {"c1": 1.0}
        y = 1.0
        update_counters(
            store=self.store,
            retrieved_sims=retrieved_sims,
            y=y,
            current_timestamp=100.0,
            gamma=1.0,
            cluster_id="cluster_test"
        )
        c1 = self.store.get_candidate("c1")
        # Global should increase (initial alpha=1.0)
        self.assertGreater(c1.alpha, 1.0)
        # Cluster-specific should also exist and increase (prior alpha starts at 1.0)
        cc = c1.get_cluster("cluster_test")
        self.assertGreater(cc["alpha"], 1.0)

    def test_recency_decay_preserves_active_doc(self):
        """Verifies that documents confirmed frequently hold their reputation, while idle ones decay."""
        c1 = Candidate(id="c_active", content="active", alpha=5.0, beta=1.0, last_confirmed=0.0, last_updated=0.0)
        c2 = Candidate(id="c_idle", content="idle", alpha=5.0, beta=1.0, last_confirmed=0.0, last_updated=0.0)
        self.store.add_candidate(c1)
        self.store.add_candidate(c2)

        # Decay unit is 1.0 sec. We do steps from t=1 to t=10.
        for t in range(1, 11):
            update_counters(
                store=self.store,
                retrieved_sims={"c_active": 1.0},
                y=1.0,
                current_timestamp=float(t),
                gamma=0.5,
                decay_unit_sec=1.0
            )

        from rrl.retriever import Retriever
        retriever = Retriever(self.store, weights=(1.0, 0.0, 0.0, 0.0))
        res = retriever.retrieve({"c_active": 1.0, "c_idle": 1.0}, top_k=2, explore=False, current_timestamp=10.0, gamma=0.5, decay_unit_sec=1.0)
        
        c_active_dec = next(c for c, _, _ in res if c.id == "c_active")
        c_idle_dec = next(c for c, _, _ in res if c.id == "c_idle")

        self.assertGreater(c_active_dec.alpha, 2.9)
        self.assertLess(c_idle_dec.alpha, 1.1)

    def test_kappa_ambiguous_outcome(self):
        """Verifies that an ambiguous outcome (y ≈ 0.5) barely moves counters."""
        y = 0.51
        update_counters(
            store=self.store,
            retrieved_sims={"c1": 1.0},
            y=y,
            current_timestamp=100.0,
            gamma=1.0
        )
        c1 = self.store.get_candidate("c1")
        self.assertLess(c1.alpha, 1.02)


if __name__ == "__main__":
    unittest.main()

