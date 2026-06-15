"""
CAG Candidate and Store Implementations
Defines the Candidate dataclass with its Beta distribution counters and the CandidateStore.
"""

from dataclasses import dataclass, field
import datetime
from typing import Dict, List, Optional


@dataclass
class Candidate:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    
    # Short-term / Recent Usefulness Counters
    alpha: float = 1.0
    beta: float = 1.0
    
    # Permanent Usefulness Counters
    A: float = 1.0
    B: float = 1.0
    
    # Robustness & Denoising Fields
    fooled: float = 0.0
    verified: float = 0.0
    recent_outcomes: List[float] = field(default_factory=list)
    
    # Timestamp tracking for decay (seconds since epoch)
    last_updated: float = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).timestamp())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "alpha": self.alpha,
            "beta": self.beta,
            "A": self.A,
            "B": self.B,
            "fooled": self.fooled,
            "verified": self.verified,
            "recent_outcomes": self.recent_outcomes,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Candidate":
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            alpha=data.get("alpha", 1.0),
            beta=data.get("beta", 1.0),
            A=data.get("A", 1.0),
            B=data.get("B", 1.0),
            fooled=data.get("fooled", 0.0),
            verified=data.get("verified", 0.0),
            recent_outcomes=data.get("recent_outcomes", []),
            last_updated=data.get("last_updated", datetime.datetime.now(datetime.timezone.utc).timestamp()),
        )


class CandidateStore:
    def __init__(self):
        self.candidates: Dict[str, Candidate] = {}

    def add_candidate(self, candidate: Candidate) -> None:
        self.candidates[candidate.id] = candidate

    def get_candidate(self, candidate_id: str, now: Optional[float] = None) -> Optional[Candidate]:
        return self.candidates.get(candidate_id)

    def list_candidates(self, now: Optional[float] = None) -> List[Candidate]:
        return list(self.candidates.values())


    def update_candidate(self, candidate: Candidate) -> None:
        if candidate.id in self.candidates:
            self.candidates[candidate.id] = candidate
        else:
            raise KeyError(f"Candidate with ID {candidate.id} not found in store.")
