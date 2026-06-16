import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from cag.store import Candidate
from cag.store_sqlite import SqliteCandidateStore
from cag.feedback import (
    OutcomeSignals,
    calculate_outcome,
    calculate_robust_estimate,
    update_counters
)


class TestRobustnessUpgrades(unittest.TestCase):
    def setUp(self):
        # Create a temp file for SQLite DB
        self.db_fd, self.db_path = tempfile.mkstemp()
        
    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_schema_migration_fresh_and_legacy(self):
        # 1. Test fresh DB initialization
        store = SqliteCandidateStore(self.db_path)
        with closing(store._connect()) as conn:
            # Check user_version is 2
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(version, 2)
            # Check table structure
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(candidates)").fetchall()]
            self.assertIn("fooled", cols)
            self.assertIn("verified", cols)
            self.assertIn("recent_outcomes", cols)

        # 2. Test legacy migration path
        # Close connection, recreate legacy DB path
        os.unlink(self.db_path)
        self.db_fd, self.db_path = tempfile.mkstemp()
        
        # Manually create candidates table with old columns only and set user_version to 0
        conn_legacy = sqlite3.connect(self.db_path)
        conn_legacy.execute("""
        CREATE TABLE candidates (
            id           TEXT PRIMARY KEY,
            content      TEXT NOT NULL,
            metadata     TEXT NOT NULL DEFAULT '{}',
            alpha        REAL NOT NULL DEFAULT 1.0,
            beta         REAL NOT NULL DEFAULT 1.0,
            A            REAL NOT NULL DEFAULT 1.0,
            B            REAL NOT NULL DEFAULT 1.0,
            last_updated REAL NOT NULL
        );
        """)
        conn_legacy.execute("PRAGMA user_version = 0")
        conn_legacy.commit()
        conn_legacy.close()

        # Load store - should run migrations and bump user_version to 2
        store_legacy = SqliteCandidateStore(self.db_path)
        with closing(store_legacy._connect()) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(version, 2)
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(candidates)").fetchall()]
            self.assertIn("fooled", cols)
            self.assertIn("verified", cols)
            self.assertIn("recent_outcomes", cols)

    def test_robust_estimators(self):
        # N=30 ring outcomes sequence
        # We need a dummy candidate to test calculate_robust_estimate
        cand = Candidate(id="c_test", content="content")
        
        # Under 10 fallback test
        cand.recent_outcomes = [0.1] * 5
        cand.alpha = 2.0
        cand.beta = 3.0
        # Should fallback to alpha/(alpha+beta) = 0.40
        self.assertAlmostEqual(calculate_robust_estimate(cand, "median"), 0.40)

        # Set 10 outcomes
        # Sorted: [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.8, 0.9, 0.9]
        cand.recent_outcomes = [0.1, 0.2, 0.9, 0.1, 0.2, 0.8, 0.3, 0.9, 0.1, 0.2]
        
        # Plain Median test (n=10, even, average of index 4 and 5: 0.2 and 0.2 -> 0.20)
        self.assertAlmostEqual(calculate_robust_estimate(cand, "median"), 0.20)

        # Trimmed Mean test (drop top 30% -> drop top 3 items [0.8, 0.9, 0.9], keep [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3] -> mean = 1.2 / 7 = 0.17142857)
        self.assertAlmostEqual(calculate_robust_estimate(cand, "trimmed"), 1.2 / 7.0)

        # Median of Means test (split into 5 blocks of size 2)
        # Block 1: [0.1, 0.2] -> mean = 0.15
        # Block 2: [0.9, 0.1] -> mean = 0.50
        # Block 3: [0.2, 0.8] -> mean = 0.50
        # Block 4: [0.3, 0.9] -> mean = 0.60
        # Block 5: [0.1, 0.2] -> mean = 0.15
        # Sorted Block Means: [0.15, 0.15, 0.50, 0.50, 0.60]
        # Median of Means: 0.50
        self.assertAlmostEqual(calculate_robust_estimate(cand, "mom"), 0.50)

    def test_trust_score_and_signals_scaling(self):
        # 1. No fooled count -> trust_score = 1.0
        signals = OutcomeSignals(s_behave=0.90, s_gt=None, s_judge=0.50, s_expl=1.0)
        # outcome = (0.45 * 0.90 + 0.15 * 0.50 + 0.10 * 1.0) / 0.70 = 0.58 / 0.70 = 0.82857
        outcome_normal = calculate_outcome(signals, cap_behave=False, gt_override=False, trust_score=1.0)
        self.assertAlmostEqual(outcome_normal, 0.58 / 0.70)

        # 2. 50% fooled -> trust_score = 0.50
        # behave scaled: 0.90 * 0.50 = 0.45
        # expl scaled: 1.0 * 0.50 = 0.50
        # outcome = (0.45 * 0.45 + 0.15 * 0.50 + 0.10 * 0.50) / 0.70 = 0.3275 / 0.70 = 0.467857
        outcome_scaled = calculate_outcome(signals, cap_behave=False, gt_override=False, trust_score=0.50)
        self.assertAlmostEqual(outcome_scaled, 0.3275 / 0.70)

    def test_adt_loss_downweighting(self):
        # Mock store with a candidate
        store = SqliteCandidateStore(self.db_path)
        cand = Candidate(id="c1", content="c1_content")
        cand.recent_outcomes = [0.4] * 12  # C_robust will be 0.4
        store.add_candidate(cand)

        # 1. Low loss step: y_credited matches C_robust (0.4) -> loss = 0.0 -> w = 1.0
        # We retrieve c1 with similarity 1.0
        # total_smoothed_sim = 1.0 + 0.0 = 1.0 -> share = 1.0 -> y_credited = y = 0.4
        update_counters(
            store=store,
            retrieved_sims={"c1": 1.0},
            y=0.4,
            current_timestamp=100.0,
            credit_smoothing=0.0,
            use_adt_denoising=True,
            robust_estimator_mode="median"
        )
        cand_updated = store.get_candidate("c1")
        # Since y = 0.4 -> kappa = 2 * |0.4 - 0.5| = 0.2 -> kappa_eff = 0.2
        # d_alpha = 0.2 * 1.0 * 0.4 = 0.08
        self.assertAlmostEqual(cand_updated.alpha, 1.08)

        # 2. High loss step: y_credited is 0.9 (huge anomaly from 0.4)
        # loss = |0.9 - 0.4| = 0.5
        # w = exp(-0.25 / 0.32) = exp(-0.78125) = 0.4578
        # kappa = 2 * |0.9 - 0.5| = 0.8
        # kappa_eff = 0.8 * 0.4578 = 0.36626
        # d_alpha = 0.36626 * 1.0 * 0.9 = 0.3296
        # alpha_before is 1.08 -> alpha_after = 1.08 + 0.3296 = 1.4096
        update_counters(
            store=store,
            retrieved_sims={"c1": 1.0},
            y=0.9,
            current_timestamp=100.0,
            credit_smoothing=0.0,
            use_adt_denoising=True,
            robust_estimator_mode="median"
        )
        cand_anomaly = store.get_candidate("c1")
        self.assertAlmostEqual(cand_anomaly.alpha, 1.08 + 0.8 * math_exp_calc(0.5) * 0.9)

    def test_update_counters_with_signals(self):
        from cag.feedback import update_counters_with_signals
        store = SqliteCandidateStore(self.db_path)
        cand = Candidate(id="c1", content="c1_content", alpha=1.0, beta=1.0, A=1.0, B=1.0, fooled=1.0, verified=2.0)
        store.add_candidate(cand)
        
        # fooled/verified = 1/2 = 0.5 -> trust_score = 0.5
        # OutcomeSignals: behave=0.90, gt=None, judge=0.50, expl=1.0
        # behave scaled: 0.90 * 0.5 = 0.45
        # expl scaled: 1.0 * 0.5 = 0.5
        # outcome = (0.45 * 0.45 + 0.15 * 0.50 + 0.10 * 0.50) / 0.70 = 0.3275 / 0.70 = 0.467857
        # y = 0.467857 -> kappa = 2 * |y - 0.5| = 2 * (0.5 - 0.467857) = 0.064286
        signals = OutcomeSignals(s_behave=0.90, s_gt=None, s_judge=0.50, s_expl=1.0)
        shares = {"c1": 0.8}
        
        # update
        update_counters_with_signals(
            store=store,
            shares=shares,
            signals=signals,
            current_timestamp=100.0,
            use_liar_counter=True,
        )
        
        cand_updated = store.get_candidate("c1")
        # since s_gt is None, fooled and verified are NOT updated
        self.assertEqual(cand_updated.fooled, 1.0)
        self.assertEqual(cand_updated.verified, 2.0)
        
        # check updates: d_alpha = kappa * share * y = 0.064286 * 0.8 * 0.467857 = 0.02406
        y = 0.3275 / 0.70
        kappa = 2.0 * (0.5 - y)
        d_alpha = kappa * 0.8 * y
        self.assertAlmostEqual(cand_updated.alpha, 1.0 + d_alpha)


def math_exp_calc(loss: float) -> float:
    import math
    return math.exp(-(loss ** 2) / 0.32)


if __name__ == "__main__":
    unittest.main()
