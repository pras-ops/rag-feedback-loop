"""
Unit Tests for CAG FastAPI App (Phase 4).
Tests /retrieve and /feedback HTTP endpoints and persistent SQLite side-effects.
"""

import os
import tempfile
import unittest
from contextlib import closing
from fastapi.testclient import TestClient

# Create a temporary SQLite database file for testing
db_file_fd, db_file_path = tempfile.mkstemp()
os.environ["CAG_DB_PATH"] = db_file_path

from cag.store import Candidate
from cag.api import app, store, retriever


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
        # Clear database candidates and pending tables before each test
        with closing(store._connect()) as conn:
            conn.execute("DELETE FROM candidates")
            conn.execute("DELETE FROM pending")
            conn.commit()

        # Mock the sentence transformer to prevent real model downloads/loading in tests
        class MockModel:
            def encode(self, text):
                class MockResult:
                    def tolist(self):
                        return [0.1] * 384
                return MockResult()
        retriever._model = MockModel()

        # Seed mock candidates
        self.c1 = Candidate(
            id="doc1",
            content="FastAPI is a modern web framework",
            metadata={"category": "tech", "embedding": [0.1] * 384},
            alpha=1.0,
            beta=1.0,
            A=1.0,
            B=1.0,
            last_updated=100.0
        )
        self.c2 = Candidate(
            id="doc2",
            content="Machine learning is subset of AI",
            metadata={"category": "ai", "embedding": [0.1] * 384},
            alpha=2.0,
            beta=3.0,
            A=2.0,
            B=3.0,
            last_updated=100.0
        )
        store.add_candidate(self.c1)
        store.add_candidate(self.c2)

    @classmethod
    def tearDownClass(cls):
        # Close and remove the temporary database file
        os.close(db_file_fd)
        if os.path.exists(db_file_path):
            os.unlink(db_file_path)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_retrieve_endpoint(self):
        payload = {
            "query": "framework",
            "top_k": 2,
            "explore": False
        }
        response = self.client.post("/retrieve", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("response_id", data)
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 2)
        
        ids = [item["candidate"]["id"] for item in data["results"]]
        self.assertIn("doc1", ids)
        self.assertIn("doc2", ids)

    def test_feedback_endpoint_success(self):
        # 1. Retrieve first to populate pending credit shares table
        payload = {
            "query": "web framework",
            "top_k": 2,
            "explore": False
        }
        retrieve_resp = self.client.post("/retrieve", json=payload)
        self.assertEqual(retrieve_resp.status_code, 200)
        retrieve_data = retrieve_resp.json()
        response_id = retrieve_data["response_id"]
        
        # 2. Submit feedback
        feedback_payload = {
            "response_id": response_id,
            "s_behave": 0.90,
            "s_gt": 1.0,
            "s_judge": 0.85,
            "s_expl": 1.0
        }
        feedback_resp = self.client.post("/feedback", json=feedback_payload)
        self.assertEqual(feedback_resp.status_code, 200)
        feedback_data = feedback_resp.json()
        self.assertEqual(feedback_data["status"], "success")
        self.assertIn("doc1", feedback_data["updated_candidates"])
        self.assertIn("doc2", feedback_data["updated_candidates"])
        
        # 3. Verify candidate stats updated in SQLite DB (s_gt=1.0 verification forces verifier increment and alpha increase)
        cand1 = store.get_candidate("doc1")
        self.assertGreater(cand1.alpha, 1.0)
        self.assertEqual(cand1.verified, 1.0)
        self.assertEqual(cand1.fooled, 0.0)

    def test_feedback_endpoint_not_found(self):
        # Submit feedback with non-existent response_id
        feedback_payload = {
            "response_id": "invalid-uuid",
            "s_behave": 0.90
        }
        feedback_resp = self.client.post("/feedback", json=feedback_payload)
        self.assertEqual(feedback_resp.status_code, 404)
        self.assertIn("not found", feedback_resp.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
