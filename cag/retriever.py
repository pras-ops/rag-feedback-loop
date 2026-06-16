"""
CAG Retriever Implementation
Implements hybrid retrieval (vector + BM25 RRF-fused), normalized sim scores,
and combines them with short-term (Beta sampled or expectation) and permanent usefulness.
Supports both real text queries (via SentenceTransformer + BM25) and pre-computed similarity scores.
"""

from collections import Counter
import math
import random
import re
from typing import Dict, List, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
from .store import Candidate, CandidateStore


def tokenize(text: str) -> List[str]:
    """Cleans punctuation, lowercases, and splits text into tokens."""
    return re.findall(r'\w+', text.lower())


class BM25:
    def __init__(self, candidates: List[Candidate], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.candidates = candidates
        self.corpus_size = len(candidates)
        
        # Tokenize doc contents
        self.doc_tokens = [tokenize(c.content) for c in candidates]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 1.0
        
        self.doc_tfs = [Counter(tokens) for tokens in self.doc_tokens]
        
        # Document frequencies (df)
        self.df: Dict[str, int] = {}
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.df[token] = self.df.get(token, 0) + 1
                
        # Precompute IDF
        self.idf: Dict[str, float] = {}
        for token, freq in self.df.items():
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query_tokens: List[str]) -> Dict[str, float]:
        scores = {}
        for i, candidate in enumerate(self.candidates):
            score = 0.0
            tf = self.doc_tfs[i]
            doc_len = self.doc_lens[i]
            for token in query_tokens:
                if token not in tf:
                    continue
                f = tf[token]
                idf_val = self.idf.get(token, 0.0)
                denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf_val * (f * (self.k1 + 1.0)) / denom
            scores[candidate.id] = score
        return scores


class Retriever:
    def __init__(
        self,
        store: CandidateStore,
        k_rrf: int = 60,
        weights: Tuple[float, float, float, float] = (0.70, 0.20, 0.10, 0.0),
        model_name: str = "all-MiniLM-L6-v2",
        model: Optional[SentenceTransformer] = None,
        robust_estimator_mode: str = "beta",
        use_optimistic_prior: bool = True,
        clusterer: Optional[object] = None,
        use_clustering: bool = True
    ):
        self.store = store
        self.k_rrf = k_rrf
        self.weights = weights
        self.model_name = model_name
        self._model = model
        self.robust_estimator_mode = robust_estimator_mode
        self.use_optimistic_prior = use_optimistic_prior
        self.use_clustering = use_clustering
        
        if clusterer is None:
            from .clustering import QueryClusterer
            self.clusterer = QueryClusterer()
            self.clusterer.load(self.store)
        else:
            self.clusterer = clusterer
        
        # BM25 index cache
        self._bm25_cached: Optional[BM25] = None
        self._bm25_candidate_ids: List[str] = []

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _compute_rrf(
        self,
        vector_scores: Dict[str, float],
        bm25_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Computes Reciprocal Rank Fusion (RRF) scores for candidates.
        """
        sorted_vector = [cid for cid, _ in sorted(vector_scores.items(), key=lambda x: x[1], reverse=True)]
        sorted_bm25 = [cid for cid, _ in sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)]

        vector_ranks = {cid: rank + 1 for rank, cid in enumerate(sorted_vector)}
        bm25_ranks = {cid: rank + 1 for rank, cid in enumerate(sorted_bm25)}

        all_cids = set(vector_scores.keys()).union(set(bm25_scores.keys()))
        rrf_scores = {}

        for cid in all_cids:
            v_rank = vector_ranks.get(cid, len(sorted_vector) + 1)
            b_rank = bm25_ranks.get(cid, len(sorted_bm25) + 1)

            v_part = 1.0 / (self.k_rrf + v_rank)
            b_part = 1.0 / (self.k_rrf + b_rank)

            rrf_scores[cid] = v_part + b_part

        return rrf_scores

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Min-max normalizes scores to the [0, 1] range.
        """
        if not scores:
            return {}
        
        vals = list(scores.values())
        min_val = min(vals)
        max_val = max(vals)
        diff = max_val - min_val

        if diff == 0:
            return {cid: 1.0 for cid in scores}

        return {cid: (val - min_val) / diff for cid, val in scores.items()}

    def retrieve(
        self,
        vector_scores: Union[str, Dict[str, float]],
        bm25_scores: Optional[Dict[str, float]] = None,
        top_k: int = 5,
        explore: bool = True,
        override_weights: Optional[Tuple[float, float, float, float]] = None,
        epsilon: float = 0.15,
        robust_estimator_mode: Optional[str] = None,
        current_timestamp: Optional[float] = None,
        gamma: float = 1.0,
        decay_unit_sec: float = 86400.0,
    ) -> List[Tuple[Candidate, float, float]]:
        """
        Retrieves the top_k candidates.
        
        Parameters:
            vector_scores: Either a real text query (str) or pre-computed vector similarity dict.
            bm25_scores: (Optional) Pre-computed BM25 similarity dict (only if first param is a dict).
            top_k: Number of candidates to retrieve.
            explore: Whether to apply Beta distribution sampling and rarity bonus.
            override_weights: Custom weights (w_sim, w_c, w_p, w_explore).
            epsilon: Epsilon-greedy parameter.
        """
        w_sim, w_c, w_p, w_explore = override_weights if override_weights is not None else self.weights

        self.last_query_cluster = "cluster_0" if self.use_clustering else None
        cluster_id = "cluster_0" if self.use_clustering else None

        # Check if first parameter is a real text query
        if isinstance(vector_scores, str):
            query = vector_scores
            candidates = self.store.list_candidates()
            if not candidates:
                return []

            # 1. Compute query embedding
            query_emb = self.model.encode(query).tolist()
            if self.use_clustering:
                cluster_id = self.clusterer.assign(query_emb)
                self.clusterer.save(self.store)
                self.last_query_cluster = cluster_id
            else:
                cluster_id = None
                self.last_query_cluster = None
            query_tokens = tokenize(query)

            # 2. Calculate vector similarity (dot product of unit length vectors)
            vector_scores = {}
            for cand in candidates:
                cand_emb = cand.metadata.get("embedding")
                if cand_emb is None:
                    # Dynamically encode and cache if missing
                    cand_emb = self.model.encode(cand.content).tolist()
                    cand.metadata["embedding"] = cand_emb
                
                # Cosine similarity
                dot_product = sum(q * c for q, c in zip(query_emb, cand_emb))
                norm_q = sum(q * q for q in query_emb) ** 0.5
                norm_c = sum(c * c for c in cand_emb) ** 0.5
                vector_scores[cand.id] = dot_product / (norm_q * norm_c) if (norm_q * norm_c) > 0 else 0.0

            # 3. Calculate BM25 scores (cached or rebuilt)
            current_cids = sorted(cand.id for cand in candidates)
            if self._bm25_cached is None or self._bm25_candidate_ids != current_cids:
                self._bm25_cached = BM25(candidates)
                self._bm25_candidate_ids = current_cids
            
            calculated_bm25_scores = self._bm25_cached.get_scores(query_tokens)
        else:
            # Pre-computed scores (backward compatible for simulation)
            calculated_bm25_scores = bm25_scores if bm25_scores is not None else {}

        # 4. Compute RRF fused scores
        rrf_scores = self._compute_rrf(vector_scores, calculated_bm25_scores)

        # 5. Normalize RRF scores to obtain sim(i) in [0, 1]
        sim_scores = self._normalize_scores(rrf_scores)

        ranked_candidates = []
        for cid, sim in sim_scores.items():
            candidate = self.store.get_candidate(cid)
            if not candidate:
                continue

            # Decay on read if not handled by SQL store
            alpha_global = candidate.alpha
            beta_global = candidate.beta
            A_global = candidate.A
            B_global = candidate.B

            if not hasattr(self.store, "increment"):
                # Recency-based decay for in-memory store: uses last_confirmed
                last_confirmed = candidate.last_confirmed
                dt = current_timestamp - last_confirmed if current_timestamp is not None else 0.0
                if dt > 0 and decay_unit_sec > 0:
                    days = dt / decay_unit_sec
                    decay_factor = gamma ** days
                    candidate.alpha = 1.0 + (candidate.alpha - 1.0) * decay_factor
                    candidate.beta = 1.0 + (candidate.beta - 1.0) * decay_factor
                    candidate.last_updated = current_timestamp
                alpha_global = candidate.alpha
                beta_global = candidate.beta

            # Apply optimistic prior for cold-start / new docs
            if self.use_optimistic_prior and (A_global + B_global <= 2.0 or (alpha_global == 1.0 and beta_global == 1.0)):
                alpha_global = 2.0

            # Hierarchical query-conditional cluster counters
            alpha_c = 1.0
            beta_c = 1.0
            A_c = 1.0
            B_c = 1.0
            n_cluster = 0.0

            cluster_counters = getattr(candidate, "cluster_counters", {})
            if cluster_id and cluster_id in cluster_counters:
                cc = cluster_counters[cluster_id]
                cc_lc = cc.get("last_confirmed", candidate.last_confirmed)
                
                # Decay cluster counters on read
                from .store_sqlite import _decay
                cc_dt = 0.0
                if current_timestamp is not None:
                    cc_dt = (current_timestamp - cc_lc) / decay_unit_sec
                
                alpha_c = _decay(cc.get("alpha", 1.0), gamma, cc_dt)
                beta_c = _decay(cc.get("beta", 1.0), gamma, cc_dt)
                A_c = cc.get("A", 1.0)
                B_c = cc.get("B", 1.0)
                n_cluster = max(0.0, A_c + B_c - 2.0)

            # Shrinkage interpolation (K_threshold = 10.0)
            N_threshold = 10.0
            lam = min(1.0, max(0.0, n_cluster / N_threshold))

            alpha = (1.0 - lam) * alpha_global + lam * alpha_c
            beta = (1.0 - lam) * beta_global + lam * beta_c
            A = (1.0 - lam) * A_global + lam * A_c
            B = (1.0 - lam) * B_global + lam * B_c

            # Robust estimation for exploitation C_robust
            from .feedback import calculate_robust_estimate
            robust_mode = robust_estimator_mode if robust_estimator_mode is not None else self.robust_estimator_mode
            C_robust = calculate_robust_estimate(candidate, robust_mode)
            if robust_mode == "beta":
                C_robust = alpha / (alpha + beta)

            # P(i) = A(i) / (A(i) + B(i))
            P_i = A / (A + B) if (A + B) > 0 else 0.5

            # Rarity/uncertainty UCB bonus with exploration floor (min 0.05)
            rarity_bonus = max(0.05, 1.0 / ((alpha + beta) ** 0.5))

            # Calculate total score using configured weights
            score = w_sim * sim + w_c * C_robust + w_p * P_i
            if explore:
                alpha_val = max(1e-5, alpha)
                beta_val = max(1e-5, beta)
                ts_sample = random.betavariate(alpha_val, beta_val)
                # Scale exploration by similarity score to prevent exploring completely irrelevant candidates
                score += w_explore * sim * (ts_sample + rarity_bonus)

            ranked_candidates.append((candidate, score, sim))


        # Sort by total score descending
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)

        results = ranked_candidates[:top_k]

        # Epsilon-greedy exploration over the full candidate set
        if explore and random.random() < epsilon and len(sim_scores) > top_k:
            all_cids = list(sim_scores.keys())
            selected_cids = [r[0].id for r in results[:top_k-1]]
            candidate_pool = [self.store.get_candidate(cid) for cid in all_cids if cid not in selected_cids]
            candidate_pool = [c for c in candidate_pool if c is not None]

            if candidate_pool:
                min_count = min(c.alpha + c.beta for c in candidate_pool)
                least_explored = [c for c in candidate_pool if (c.alpha + c.beta) <= min_count + 1e-5]
                explorer_cand = random.choice(least_explored)

                explorer_tuple = None
                for r_tuple in ranked_candidates:
                    if r_tuple[0].id == explorer_cand.id:
                        explorer_tuple = r_tuple
                        break

                if explorer_tuple:
                    results = results[:top_k-1] + [explorer_tuple]

        return results
