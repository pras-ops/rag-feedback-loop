import unittest
import os
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cag.store import Candidate, CandidateStore
from cag.retriever import Retriever
from cag.clustering import QueryClusterer


import numpy as np

# Mock sentence transformer to avoid network requests in unit tests
class MockSentenceTransformer:
    def encode(self, text):
        val = sum(ord(c) for c in text) % 100 / 100.0
        return np.array([val, 1.0 - val, 0.0])


class TestRetrieverLogic(unittest.TestCase):
    def test_shrinkage_thin_cluster_uses_global(self):
        store = CandidateStore()
        # candidate alpha_global = 5.0, beta_global = 2.0
        cand = Candidate(id="d1", content="test doc", alpha=5.0, beta=2.0, A=3.0, B=2.0)
        # Cluster has 0 observations
        store.add_candidate(cand)
        
        # Instantiate Retriever with weights highlighting permanent & short usefulness
        retriever = Retriever(store, weights=(0.0, 0.5, 0.5, 0.0), model=MockSentenceTransformer())
        
        # Trigger retrieve (which assigns to cluster_0)
        # cluster_0 has 0 observations. Thus lam = 0.0, alpha_eff should equal alpha_global = 5.0
        # Since we use weights=(0.0, 0.5, 0.5, 0.0) -> score = 0.5 * C_robust + 0.5 * P_i
        # For d1: C_robust = 5 / (5 + 2) = 5/7 = 0.714
        # P_i = A / (A + B) = 3 / (3 + 2) = 0.6
        # expected score = 0.5 * 0.714 + 0.5 * 0.6 = 0.657
        res = retriever.retrieve("query", top_k=1, explore=False)
        self.assertEqual(res[0][0].id, "d1")
        self.assertAlmostEqual(res[0][1], 0.5 * (5.0/7.0) + 0.5 * (3.0/5.0))

    def test_shrinkage_rich_cluster_uses_local(self):
        store = CandidateStore()
        # candidate alpha_global = 5.0, beta_global = 2.0, A_global=3.0, B_global=2.0
        # cluster counter alpha=10.0, beta=10.0, A=10.0, B=10.0
        cand = Candidate(id="d1", content="test doc", alpha=5.0, beta=2.0, A=3.0, B=2.0)
        # Set cluster_0 with 18 observations (A_c=10.0, B_c=10.0 => n = 10+10-2 = 18 >= 10.0 => lam = 1.0)
        cand.cluster_counters["cluster_0"] = {
            "alpha": 10.0,
            "beta": 10.0,
            "A": 10.0,
            "B": 10.0,
            "fooled": 0.0,
            "verified": 0.0,
            "recent_outcomes": [],
            "last_confirmed": 0.0
        }
        store.add_candidate(cand)
        
        retriever = Retriever(store, weights=(0.0, 0.5, 0.5, 0.0), model=MockSentenceTransformer())
        
        # Retrieve: lam = 1.0 => alpha_eff = 10.0, beta_eff = 10.0, A_eff = 10.0, B_eff = 10.0
        # C_robust = 10 / 20 = 0.5
        # P_i = 10 / 20 = 0.5
        # expected score = 0.5 * 0.5 + 0.5 * 0.5 = 0.5
        res = retriever.retrieve("query", top_k=1, explore=False)
        self.assertAlmostEqual(res[0][1], 0.5)

    def test_optimistic_prior_new_doc(self):
        store = CandidateStore()
        # d1 is established: alpha=10.0, beta=10.0 (low uncertainty, robust estimate ~ 0.5)
        d1 = Candidate(id="d1", content="established", alpha=10.0, beta=10.0, A=10.0, B=10.0)
        # d2 is brand new: alpha=1.0, beta=1.0, but gets optimistic alpha=2.0, beta=1.0
        d2 = Candidate(id="d2", content="new doc", alpha=1.0, beta=1.0, A=1.0, B=1.0)
        store.add_candidate(d1)
        store.add_candidate(d2)

        # RRF similarity weight is 0.0, we prioritize C_robust (0.0, 1.0, 0.0, 0.0)
        retriever = Retriever(store, weights=(0.0, 1.0, 0.0, 0.0), model=MockSentenceTransformer(), use_optimistic_prior=True)
        res = retriever.retrieve("query", top_k=2, explore=False)
        # Because of optimistic prior, d2 (new) has C_robust = 2/(2+1) = 0.667
        # d1 (established) has C_robust = 10/(10+10) = 0.500
        # So d2 should be retrieved first!
        self.assertEqual(res[0][0].id, "d2")
        self.assertEqual(res[1][0].id, "d1")

    def test_explore_floor_prevents_starvation(self):
        store = CandidateStore()
        d1 = Candidate(id="d1", content="highly active", alpha=100.0, beta=1.0, A=100.0, B=1.0)
        d2 = Candidate(id="d2", content="less active", alpha=1.0, beta=1.0, A=1.0, B=1.0)
        store.add_candidate(d1)
        store.add_candidate(d2)

        retriever = Retriever(store, weights=(0.0, 0.0, 0.0, 1.0), model=MockSentenceTransformer())
        
        # Run multiple trials and assert d2 is retrieved at least once
        retrieved_d2 = False
        random.seed(42)
        for _ in range(50):
            res = retriever.retrieve("query", top_k=1, explore=True)
            if res[0][0].id == "d2":
                retrieved_d2 = True
                break
        self.assertTrue(retrieved_d2)


if __name__ == "__main__":
    unittest.main()
