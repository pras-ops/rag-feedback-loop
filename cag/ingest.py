"""
CAG Ingestion Module
Handles document chunking and vector embedding generation using sentence-transformers.
"""

from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from .store import Candidate, CandidateStore


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Splits text into chunks using a character-level sliding window.
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move start forward by step (chunk_size - overlap)
        start += (chunk_size - overlap)
        
        # Prevent infinite loops if step is 0 or negative
        if chunk_size <= overlap:
            break
            
    return chunks


class Ingester:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load the local sentence-transformer model (cached once)
        self.model = SentenceTransformer(model_name)

    def ingest_document(
        self,
        store: CandidateStore,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Chunks the document text, encodes each chunk to a vector embedding,
        creates Candidate objects, and registers them in the CandidateStore.
        
        Returns:
            List of generated candidate IDs.
        """
        chunks = chunk_text(text)
        candidate_ids = []

        # Bulk encode chunks for performance
        if not chunks:
            return []
            
        embeddings = self.model.encode(chunks)

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            candidate_id = f"{doc_id}_chunk_{i}"
            
            # Pack embedding list and other attributes in metadata
            cand_metadata = {
                "doc_id": doc_id,
                "chunk_idx": i,
                "embedding": emb.tolist(),
                **(metadata or {})
            }

            candidate = Candidate(
                id=candidate_id,
                content=chunk,
                metadata=cand_metadata,
                alpha=1.0,
                beta=1.0,
                A=1.0,
                B=1.0
            )

            store.add_candidate(candidate)
            candidate_ids.append(candidate_id)

        return candidate_ids
