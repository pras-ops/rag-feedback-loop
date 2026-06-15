"""
Phase 4 store validation. Runnable directly: `python3 tests/test_store_sqlite.py`
Proves: durability across reconnect, lazy decay on update, atomic concurrent
increments, and the pending retrieve<->feedback bridge.
"""

import os
import sys
import tempfile
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cag.store import Candidate
from cag.store_sqlite import SqliteCandidateStore


def _fresh_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)  # let sqlite create it
    return path


def test_durability_across_reconnect():
    path = _fresh_db()
    try:
        s1 = SqliteCandidateStore(path)
        s1.add_candidate(Candidate(id="d1", content="hello", alpha=3.0, beta=2.0, last_updated=100.0))
        del s1
        s2 = SqliteCandidateStore(path)  # new connection / fresh process simulation
        c = s2.get_candidate("d1")
        assert c is not None, "candidate did not survive reconnect"
        assert c.alpha == 3.0 and c.beta == 2.0, f"counters not durable: {c.alpha},{c.beta}"
        print("PASS durability_across_reconnect")
    finally:
        os.remove(path)


def test_lazy_decay_on_increment():
    path = _fresh_db()
    try:
        # gamma 0.5/day: after 1 day, (alpha-1) halves.
        s = SqliteCandidateStore(path, gamma=0.5, decay_unit_sec=1.0)
        s.add_candidate(Candidate(id="d1", content="x", alpha=5.0, beta=1.0, last_updated=0.0))
        # Increment 1 unit later with zero delta -> alpha should decay 5 -> 1+(5-1)*0.5 = 3.0
        s.increment("d1", d_alpha=0.0, d_beta=0.0, d_A=0.0, d_B=0.0, now=1.0)
        c = s.get_candidate("d1")
        assert abs(c.alpha - 3.0) < 1e-9, f"decay wrong: got {c.alpha}, expected 3.0"
        print("PASS lazy_decay_on_increment")
    finally:
        os.remove(path)


def test_atomic_concurrent_increments():
    path = _fresh_db()
    try:
        # gamma=1.0 disables decay so we can check exact sums.
        s = SqliteCandidateStore(path, gamma=1.0)
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
        assert abs(c.alpha - expected) < 1e-9, f"LOST UPDATES: got {c.alpha}, expected {expected}"
        print(f"PASS atomic_concurrent_increments (alpha={c.alpha}, expected {expected})")
    finally:
        os.remove(path)


def test_pending_bridge():
    path = _fresh_db()
    try:
        s = SqliteCandidateStore(path)
        s.save_pending("resp-1", {"d1": 0.7, "d2": 0.3}, now=100.0)
        shares = s.pop_pending("resp-1")
        assert shares == {"d1": 0.7, "d2": 0.3}, f"shares wrong: {shares}"
        assert s.pop_pending("resp-1") is None, "pending not consumed (double-spend!)"
        # GC
        s.save_pending("resp-old", {"d1": 1.0}, now=0.0)
        deleted = s.gc_pending(max_age_sec=10.0, now=1000.0)
        assert deleted == 1, f"gc deleted {deleted}, expected 1"
        print("PASS pending_bridge")
    finally:
        os.remove(path)


if __name__ == "__main__":
    test_durability_across_reconnect()
    test_lazy_decay_on_increment()
    test_atomic_concurrent_increments()
    test_pending_bridge()
    print("\nAll Phase 4 store tests passed.")
