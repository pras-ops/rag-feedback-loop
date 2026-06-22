"""
Phase 4 SQLite CandidateStore validation.
Proves: durability across reconnect, lazy decay on update, atomic concurrent
increments, and the pending retrieve<->feedback bridge.
"""

import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rrl.store import Candidate
from rrl.store_sqlite import SqliteCandidateStore


class TestSqliteCandidateStore(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.remove(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_durability_across_reconnect(self):
        s1 = SqliteCandidateStore(self.db_path)
        s1.add_candidate(Candidate(id="d1", content="hello", alpha=3.0, beta=2.0, last_updated=100.0))
        del s1
        s2 = SqliteCandidateStore(self.db_path)  # new connection / fresh process simulation
        c = s2.get_candidate("d1")
        self.assertIsNotNone(c, "candidate did not survive reconnect")
        self.assertEqual(c.alpha, 3.0)
        self.assertEqual(c.beta, 2.0)

    def test_lazy_decay_on_increment(self):
        # gamma 0.5/day: after 1 day, (alpha-1) halves.
        s = SqliteCandidateStore(self.db_path, gamma=0.5, decay_unit_sec=1.0)
        s.add_candidate(Candidate(id="d1", content="x", alpha=5.0, beta=1.0, last_confirmed=0.0, last_updated=0.0))
        # Increment 1 unit later with zero delta -> alpha should decay 5 -> 1+(5-1)*0.5 = 3.0
        s.increment("d1", d_alpha=0.0, d_beta=0.0, d_A=0.0, d_B=0.0, now=1.0)
        c = s.get_candidate("d1")
        self.assertAlmostEqual(c.alpha, 3.0)

    def test_atomic_concurrent_increments(self):
        # gamma=1.0 disables decay so we can check exact sums.
        s = SqliteCandidateStore(self.db_path, gamma=1.0)
        s.add_candidate(Candidate(id="d1", content="x", alpha=1.0, beta=1.0, last_updated=0.0))

        n_threads, per_thread = 8, 200
        def worker():
            for _ in range(per_thread):
                s.increment("d1", d_alpha=1.0, d_beta=0.0, d_A=0.0, d_B=0.0, now=0.0)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        c = s.get_candidate("d1")
        expected = 1.0 + n_threads * per_thread  # 1 + 1600
        self.assertAlmostEqual(c.alpha, expected)

    def test_pending_bridge(self):
        s = SqliteCandidateStore(self.db_path)
        s.save_pending("resp-1", {"d1": 0.7, "d2": 0.3}, now=100.0)
        res = s.pop_pending("resp-1")
        self.assertIsNotNone(res)
        shares, cluster_id = res
        self.assertEqual(shares, {"d1": 0.7, "d2": 0.3})
        self.assertIsNone(cluster_id)
        self.assertIsNone(s.pop_pending("resp-1"), "pending not consumed (double-spend!)")
        
        # GC
        s.save_pending("resp-old", {"d1": 1.0}, now=0.0)
        deleted = s.gc_pending(max_age_sec=10.0, now=1000.0)
        self.assertEqual(deleted, 1)

    def test_schema_v2_migration(self):
        # Create legacy v0 database manually
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "CREATE TABLE candidates ("
            "id TEXT PRIMARY KEY, "
            "content TEXT, "
            "metadata TEXT, "
            "alpha REAL, "
            "beta REAL, "
            "A REAL, "
            "B REAL, "
            "last_updated REAL"
            ")"
        )
        conn.execute("INSERT INTO candidates VALUES ('legacy_d', 'legacy content', '{}', 1.0, 1.0, 1.0, 1.0, 100.0)")
        conn.commit()
        conn.close()

        # Instantiate store - triggers upgrade migrations to v1 and v2
        s = SqliteCandidateStore(self.db_path)
        
        # Verify schema version
        conn = s._connect()
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(version, 2)
        finally:
            conn.close()
            
        # Verify legacy candidate loads with appropriate v2 defaults
        c = s.get_candidate("legacy_d")
        self.assertIsNotNone(c)
        self.assertEqual(c.content, "legacy content")
        self.assertEqual(c.cluster_counters, {})
        self.assertEqual(c.fooled, 0.0)
        self.assertEqual(c.verified, 0.0)

    def test_cluster_counter_atomic_increment(self):
        s = SqliteCandidateStore(self.db_path, gamma=1.0)
        s.add_candidate(Candidate(id="d1", content="x", alpha=1.0, beta=1.0, last_updated=0.0))

        n_threads, per_thread = 8, 200
        def worker():
            for _ in range(per_thread):
                s.increment("d1", d_alpha=1.0, d_beta=0.0, d_A=0.0, d_B=0.0, cluster_id="cluster_test", now=0.0)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        c = s.get_candidate("d1")
        cc = c.get_cluster("cluster_test")
        expected = 1.0 + n_threads * per_thread  # 1 + 1600
        self.assertAlmostEqual(cc["alpha"], expected)

    def test_last_confirmed_set_on_positive(self):
        s = SqliteCandidateStore(self.db_path)
        s.add_candidate(Candidate(id="d1", content="x", alpha=1.0, beta=1.0, last_confirmed=100.0, last_updated=100.0))
        
        # Positive outcome updates last_confirmed
        s.increment("d1", d_alpha=0.0, d_beta=0.0, d_A=0.0, d_B=0.0, recent_outcome=1.0, now=200.0)
        c = s.get_candidate("d1")
        self.assertEqual(c.last_confirmed, 200.0)

        # Negative outcome does not update last_confirmed
        s.increment("d1", d_alpha=0.0, d_beta=0.0, d_A=0.0, d_B=0.0, recent_outcome=0.0, now=300.0)
        c = s.get_candidate("d1")
        self.assertEqual(c.last_confirmed, 200.0)

    def test_pending_roundtrip_with_cluster(self):
        s = SqliteCandidateStore(self.db_path)
        s.save_pending("resp-2", {"d1": 0.5}, cluster_id="cluster_test", now=100.0)
        shares, cluster_id = s.pop_pending("resp-2")
        self.assertEqual(shares, {"d1": 0.5})
        self.assertEqual(cluster_id, "cluster_test")


if __name__ == "__main__":
    unittest.main()
