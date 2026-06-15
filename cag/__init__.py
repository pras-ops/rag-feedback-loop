"""
CAG Core Logic Package
RAG Feedback and Exploration Loop implementation.
"""

from .store import Candidate, CandidateStore
from .retriever import Retriever
from .feedback import OutcomeSignals, calculate_outcome, update_counters, update_counters_from_shares, update_counters_with_signals

__all__ = [
    "Candidate",
    "CandidateStore",
    "Retriever",
    "OutcomeSignals",
    "calculate_outcome",
    "update_counters",
    "update_counters_from_shares",
    "update_counters_with_signals",
]
