import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rrl.clustering import QueryClusterer
from rrl.store import CandidateStore


class TestQueryClustering(unittest.TestCase):
    def test_same_family_same_cluster(self):
        clusterer = QueryClusterer()
        c1 = clusterer.assign([1.0, 0.0, 0.0])
        # Near identical vector
        c2 = clusterer.assign([0.99, 0.01, 0.0])
        self.assertEqual(c1, c2)
        self.assertEqual(c1, "cluster_0")

    def test_distinct_families_split(self):
        clusterer = QueryClusterer()
        c1 = clusterer.assign([1.0, 0.0, 0.0])
        # Orthogonal vector (similarity 0.0 < tau 0.45)
        c2 = clusterer.assign([0.0, 1.0, 0.0])
        self.assertNotEqual(c1, c2)
        self.assertEqual(c1, "cluster_0")
        self.assertEqual(c2, "cluster_1")

    def test_max_clusters_cap(self):
        clusterer = QueryClusterer()
        # Feed 15 distinct orthogonal vectors to verify the cap of 10
        for i in range(15):
            vec = [0.0] * 15
            vec[i] = 1.0
            clusterer.assign(vec, max_clusters=10, tau=0.45)
        self.assertEqual(len(clusterer.centroids), 10)

    def test_centroid_persistence(self):
        store = CandidateStore()
        clusterer = QueryClusterer()
        clusterer.assign([1.0, 0.0, 0.0])
        clusterer.assign([0.0, 1.0, 0.0])
        clusterer.save(store)

        # Load into a new clusterer instance
        new_clusterer = QueryClusterer()
        new_clusterer.load(store)
        self.assertEqual(len(new_clusterer.centroids), 2)
        
        # Test assign vector close to cluster 1
        c = new_clusterer.assign([0.05, 0.95, 0.0])
        self.assertEqual(c, "cluster_1")


if __name__ == "__main__":
    unittest.main()
