import json
import numpy as np
from typing import List, Dict, Tuple, Optional

class QueryClusterer:
    def __init__(self, centroids: Optional[List[List[float]]] = None, counts: Optional[List[int]] = None):
        self.centroids = [np.array(c, dtype=float) for c in centroids] if centroids else []
        self.counts = list(counts) if counts else [1] * len(self.centroids)

    def assign(self, query_emb: List[float], max_clusters: int = 10, tau: float = 0.45) -> str:
        """
        Assigns the query embedding to the nearest centroid. 
        Updates the centroid running mean if similarity >= tau.
        Spawns a new centroid if similarity < tau and count < max_clusters.
        """
        x = np.array(query_emb, dtype=float)
        norm_x = np.linalg.norm(x)
        if norm_x > 0.0:
            x /= norm_x

        if not self.centroids:
            self.centroids.append(x)
            self.counts.append(1)
            return "cluster_0"

        best_idx = -1
        best_sim = -1.0
        for i, c in enumerate(self.centroids):
            # Centroids and query are unit length, similarity is dot product
            sim = float(np.dot(x, c))
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim >= tau:
            self.counts[best_idx] += 1
            c = self.centroids[best_idx]
            # Online centroid update: c <- c + (x - c)/count
            c += (x - c) / self.counts[best_idx]
            norm_c = np.linalg.norm(c)
            if norm_c > 0.0:
                c /= norm_c
            self.centroids[best_idx] = c
            return f"cluster_{best_idx}"
        elif len(self.centroids) < max_clusters:
            self.centroids.append(x)
            self.counts.append(1)
            return f"cluster_{len(self.centroids) - 1}"
        else:
            # Cap reached; assign to best anyway
            self.counts[best_idx] += 1
            c = self.centroids[best_idx]
            c += (x - c) / self.counts[best_idx]
            norm_c = np.linalg.norm(c)
            if norm_c > 0.0:
                c /= norm_c
            self.centroids[best_idx] = c
            return f"cluster_{best_idx}"

    def save(self, store) -> None:
        """Saves centroids and counts to store's settings table."""
        data = {
            "centroids": [c.tolist() for c in self.centroids],
            "counts": self.counts
        }
        if hasattr(store, "save_setting"):
            try:
                store.save_setting("cluster_centroids", json.dumps(data))
            except Exception:
                pass
        else:
            # Fallback for in-memory store
            store.settings = getattr(store, "settings", {})
            store.settings["cluster_centroids"] = json.dumps(data)

    def load(self, store) -> None:
        """Loads centroids and counts from store's settings table."""
        raw = None
        if hasattr(store, "get_setting"):
            try:
                raw = store.get_setting("cluster_centroids")
            except Exception:
                pass
        else:
            settings = getattr(store, "settings", {})
            raw = settings.get("cluster_centroids")

        if raw:
            try:
                data = json.loads(raw)
                self.centroids = [np.array(c, dtype=float) for c in data["centroids"]]
                self.counts = list(data["counts"])
            except Exception:
                pass


def get_query_cluster(store, query_emb: List[float], max_clusters: int = 10) -> str:
    """Backward-compatible wrapper for get_query_cluster using QueryClusterer."""
    clusterer = QueryClusterer()
    clusterer.load(store)
    cid = clusterer.assign(query_emb, max_clusters=max_clusters)
    clusterer.save(store)
    return cid


def cluster_report(history: List[Tuple[str, float]]) -> dict:
    """
    Computes clustering diagnostics:
    - avg queries per cluster
    - within-cluster outcome agreement (fraction matching the cluster's majority outcome)
    """
    if not history:
        return {
            "avg_queries_per_cluster": 0.0,
            "within_cluster_agreement": 1.0,
            "cluster_counts": {}
        }
    
    # Map cluster_id to list of outcomes
    cluster_outcomes = {}
    for cid, outcome in history:
        if cid not in cluster_outcomes:
            cluster_outcomes[cid] = []
        cluster_outcomes[cid].append(outcome)
        
    num_clusters = len(cluster_outcomes)
    avg_queries = len(history) / num_clusters if num_clusters > 0 else 0.0
    
    # Within-cluster outcome agreement
    total_matching = 0
    for cid, outcomes in cluster_outcomes.items():
        ones = sum(1 for o in outcomes if o > 0.5)
        zeros = len(outcomes) - ones
        majority_count = max(ones, zeros)
        total_matching += majority_count
        
    agreement = total_matching / len(history) if len(history) > 0 else 1.0
    
    return {
        "avg_queries_per_cluster": avg_queries,
        "within_cluster_agreement": agreement,
        "cluster_counts": {cid: len(outcomes) for cid, outcomes in cluster_outcomes.items()}
    }
