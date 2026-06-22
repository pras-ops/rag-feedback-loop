"""
RRL LLM Judge Module
Integrates google-genai to evaluate faithfulness of retrieved chunks.
Provides a local token-overlap fallback when offline or unauthenticated.
Includes per-call timeout (30s) and single retry with backoff.
"""

import os
import re
import concurrent.futures
import time
from typing import Optional
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Module-level client singleton (avoid re-creating per call)
_CLIENT: Optional[object] = None
_CLIENT_INITIALIZED = False

# Timeout per API call in seconds
_CALL_TIMEOUT_SEC = 30


class FaithfulnessRating(BaseModel):
    score: float = Field(
        description="Faithfulness score from 0.0 to 1.0. 1.0 means the answer is fully supported by the document chunk with no hallucination or mismatch. 0.0 means completely unsupported or contradicted."
    )
    reason: str = Field(
        description="A brief explanation for the assigned score."
    )


def _heuristic_fallback_judge(document_chunk: str, generated_answer: str) -> float:
    """
    Fallback similarity/overlap judge when Vertex AI / Gemini API is unavailable.
    Measures overlap of important keywords between document chunk and generated answer.
    """
    def tokenize(text: str) -> set:
        tokens = re.findall(r'\w+', text.lower())
        # Filter out common short stopwords
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "of", "is", "are"}
        return {t for t in tokens if len(t) > 2 and t not in stopwords}

    doc_tokens = tokenize(document_chunk)
    ans_tokens = tokenize(generated_answer)
    
    if not ans_tokens:
        return 0.5
        
    overlap = ans_tokens.intersection(doc_tokens)
    # Return percentage of answer keywords backed by the document
    score = len(overlap) / len(ans_tokens)
    return max(0.1, min(1.0, score))


def _get_client(project_id: Optional[str] = None):
    """Returns a cached genai Client, creating one on first call."""
    global _CLIENT, _CLIENT_INITIALIZED
    if _CLIENT_INITIALIZED:
        return _CLIENT

    _CLIENT_INITIALIZED = True

    if not GENAI_AVAILABLE:
        return None

    # 1. Try Vertex AI client first
    gcp_project = os.environ.get("GCP_PROJECT_ID", project_id)
    if gcp_project:
        try:
            gcp_region = os.environ.get("GCP_REGION", "us-central1")
            _CLIENT = genai.Client(vertexai=True, project=gcp_project, location=gcp_region)
            return _CLIENT
        except Exception:
            pass

    # 2. Try standard API Key client fallback
    if os.environ.get("GEMINI_API_KEY"):
        try:
            _CLIENT = genai.Client()
            return _CLIENT
        except Exception:
            pass

    return None


def evaluate_faithfulness(
    query: str,
    document_chunk: str,
    generated_answer: str,
    project_id: Optional[str] = None,
    max_retries: int = 1,
) -> float:
    """
    Evaluates the faithfulness of a generated answer against a candidate document chunk.
    Uses Vertex AI Gemini with a per-call timeout and retry, falling back to heuristic overlap.
    """
    client = _get_client(project_id)
    if client is None:
        return _heuristic_fallback_judge(document_chunk, generated_answer)

    prompt = (
        "You are a RAG faithfulness judge. Given the user query, a reference document chunk, "
        "and a generated answer, you must rate whether the answer is faithful to the reference document.\n"
        "The answer should not contain factual claims or assertions that are not present in or "
        "supported by the reference document chunk.\n\n"
        f"User Query: {query}\n"
        f"Reference Document: {document_chunk}\n"
        f"Generated Answer: {generated_answer}\n"
    )

    def _invoke():
        return client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FaithfulnessRating,
                temperature=0.0,
            ),
        )

    # Portable per-call timeout: run the blocking SDK call in a worker thread and
    # bound it with future.result(timeout=...). Unlike signal.SIGALRM this works
    # off the main thread (e.g. FastAPI's threadpool) and on Windows.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        for attempt in range(max_retries + 1):
            try:
                response = executor.submit(_invoke).result(timeout=_CALL_TIMEOUT_SEC)
                rating: FaithfulnessRating = response.parsed
                return max(0.0, min(1.0, rating.score))

            except concurrent.futures.TimeoutError:
                print(f"[Judge Warning] Gemini call timed out (attempt {attempt + 1}/{max_retries + 1}).")
            except Exception as e:
                print(f"[Judge Warning] Gemini call failed ({e}) (attempt {attempt + 1}/{max_retries + 1}).")

            # Backoff before retry
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    finally:
        # Do not block on a still-running (timed-out) call.
        executor.shutdown(wait=False)

    # All retries exhausted — fall back to heuristic
    print("[Judge] Falling back to heuristic overlap judge.")
    return _heuristic_fallback_judge(document_chunk, generated_answer)
