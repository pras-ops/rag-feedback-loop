"""
CAG FastAPI Application (Phase 4).
Exposes POST /retrieve and POST /feedback endpoints.
"""

import os
import uuid
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from cag.store import Candidate
from cag.store_sqlite import SqliteCandidateStore
from cag.retriever import Retriever
from cag.feedback import OutcomeSignals, update_counters_with_signals

app = FastAPI(
    title="CAG Feedback & Exploration Loop API",
    description="API for Phase 4 persistent store retrieval and feedback updates",
    version="1.0.0"
)

# Configuration from env variables
DB_PATH = os.getenv("CAG_DB_PATH", "cag.db")
DECAY_UNIT_SEC = float(os.getenv("CAG_DECAY_UNIT_SEC", "86400.0"))
GAMMA = float(os.getenv("CAG_GAMMA", "0.98"))

# Initialize components (lazily initialized or created at startup)
store = SqliteCandidateStore(db_path=DB_PATH, gamma=GAMMA, decay_unit_sec=DECAY_UNIT_SEC)
# Default weights to balanced exploration: (0.20, 0.40, 0.10, 0.30)
retriever = Retriever(store, weights=(0.20, 0.40, 0.10, 0.30))


# Pydantic schemas
class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Query string for semantic/keyword retrieval")
    top_k: int = Field(5, ge=1, description="Number of top candidates to retrieve")
    explore: bool = Field(True, description="Whether to apply exploration sampling and rarity bonus")


class CandidateSchema(BaseModel):
    id: str
    content: str
    metadata: dict
    alpha: float
    beta: float
    A: float
    B: float
    fooled: float
    verified: float
    recent_outcomes: List[float]
    cluster_counters: dict = {}
    last_confirmed: float
    last_updated: float


class RetrievalItem(BaseModel):
    candidate: CandidateSchema
    score: float
    similarity: float


class RetrieveResponse(BaseModel):
    response_id: str
    results: List[RetrievalItem]


class FeedbackRequest(BaseModel):
    response_id: str = Field(..., description="Unique ID returned from the /retrieve call")
    s_behave: Optional[float] = Field(None, ge=0.0, le=1.0, description="Behavioral keep/edit/regen score")
    s_gt: Optional[float] = Field(None, ge=0.0, le=1.0, description="Ground truth verification score")
    s_judge: Optional[float] = Field(None, ge=0.0, le=1.0, description="LLM judge score")
    s_expl: Optional[float] = Field(None, ge=0.0, le=1.0, description="Explicit thumbs-up/down score")


@app.get("/health")
def health():
    return {"status": "ok", "db_path": DB_PATH}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    try:
        # 1. Retrieve candidates
        results = retriever.retrieve(
            vector_scores=req.query,
            top_k=req.top_k,
            explore=req.explore
        )
        
        # 2. Extract retrieved candidate sims
        retrieved_sims = {r[0].id: r[2] for r in results}
        
        # 3. Calculate credit shares r(i) with smoothing
        credit_smoothing = 0.10
        total_smoothed_sim = sum(sim + credit_smoothing for sim in retrieved_sims.values())
        shares = {}
        if total_smoothed_sim > 0.0:
            for cid, sim in retrieved_sims.items():
                shares[cid] = (sim + credit_smoothing) / total_smoothed_sim
        elif retrieved_sims:
            share = 1.0 / len(retrieved_sims)
            for cid in retrieved_sims:
                shares[cid] = share
                
        # 4. Save pending shares mapped to generated response_id
        response_id = str(uuid.uuid4())
        if shares:
            cluster_id = getattr(retriever, "last_query_cluster", "cluster_0")
            store.save_pending(response_id, shares, cluster_id)
            
        # 5. Format results response
        items = []
        for cand, score, sim in results:
            items.append(
                RetrievalItem(
                    candidate=CandidateSchema(
                        id=cand.id,
                        content=cand.content,
                        metadata=cand.metadata,
                        alpha=cand.alpha,
                        beta=cand.beta,
                        A=cand.A,
                        B=cand.B,
                        fooled=cand.fooled,
                        verified=cand.verified,
                        recent_outcomes=cand.recent_outcomes,
                        cluster_counters=cand.cluster_counters,
                        last_confirmed=cand.last_confirmed,
                        last_updated=cand.last_updated
                    ),
                    score=score,
                    similarity=sim
                )
            )
            
        return RetrieveResponse(response_id=response_id, results=items)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}"
        )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    # 1. Pop pending credit shares
    res = store.pop_pending(req.response_id)
    if res is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pending shares for the given response_id not found or already processed."
        )
    shares, cluster_id = res
        
    # 2. Build signals
    signals = OutcomeSignals(
        s_behave=req.s_behave,
        s_gt=req.s_gt,
        s_judge=req.s_judge,
        s_expl=req.s_expl
    )
    
    # 3. Update candidate counters using the popped shares
    try:
        update_counters_with_signals(
            store=store,
            shares=shares,
            signals=signals,
            use_liar_counter=True,
            use_adt_denoising=False,
            robust_estimator_mode="beta",
            cluster_id=cluster_id
        )
        return {"status": "success", "updated_candidates": list(shares.keys())}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback processing failed: {str(e)}"
        )
