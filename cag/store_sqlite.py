"""
Persistent SQLite-backed CandidateStore (Phase 4).

Drop-in for the in-memory CandidateStore, plus the three things production needs
that the simulation never had to face:

  1. Durability  - counters survive a process restart.
  2. Lazy decay  - gamma^dt is applied at the moment of update, computed from
                   last_updated, so there is no cron sweep.
  3. Atomic increments - two concurrent feedback calls both land, because the
                   read-decay-add-write happens inside a single BEGIN IMMEDIATE
                   transaction instead of read-modify-write in Python.

Also adds the `pending` table: in production, /retrieve and /feedback are
separate requests, so the frozen credit shares r(i) must be persisted at
retrieval time and looked up when feedback arrives.
"""

import json
import sqlite3
import time
from typing import Dict, List, Optional

from .store import Candidate


SCHEMA_MODERN = """
CREATE TABLE IF NOT EXISTS candidates (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    metadata        TEXT NOT NULL DEFAULT '{}',
    alpha           REAL NOT NULL DEFAULT 1.0,
    beta            REAL NOT NULL DEFAULT 1.0,
    A               REAL NOT NULL DEFAULT 1.0,
    B               REAL NOT NULL DEFAULT 1.0,
    fooled          REAL NOT NULL DEFAULT 0.0,
    verified        REAL NOT NULL DEFAULT 0.0,
    recent_outcomes TEXT NOT NULL DEFAULT '[]',
    last_updated    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS pending (
    response_id TEXT PRIMARY KEY,
    shares      TEXT NOT NULL,   -- JSON {candidate_id: credit_share}
    created     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    agree     INTEGER NOT NULL DEFAULT 0,
    disagree  INTEGER NOT NULL DEFAULT 0
);
"""


def _decay(value: float, gamma: float, dt_units: float) -> float:
    """Beta-counter decay toward the prior of 1.0:  x <- 1 + (x-1) * gamma^dt."""
    if gamma >= 1.0 or dt_units <= 0:
        return value
    return 1.0 + (value - 1.0) * (gamma ** dt_units)


class SqliteCandidateStore:
    def __init__(
        self,
        db_path: str,
        gamma: float = 1.0,
        decay_unit_sec: float = 86400.0,
    ):
        self.db_path = db_path
        self.gamma = gamma
        self.decay_unit_sec = decay_unit_sec
        
        with self._connect() as conn:
            # Check if candidates table exists
            table_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='candidates'"
            ).fetchone()
            
            if not table_exists:
                # Fresh DB: create tables using modern schema and set user_version=1
                conn.executescript(SCHEMA_MODERN)
                conn.execute("PRAGMA user_version = 1")
            else:
                # Existing DB: check user_version
                version_row = conn.execute("PRAGMA user_version").fetchone()
                user_version = version_row[0] if version_row else 0
                
                if user_version < 1:
                    # Run schema migrations to match modern schema
                    conn.execute("ALTER TABLE candidates ADD COLUMN fooled REAL NOT NULL DEFAULT 0.0")
                    conn.execute("ALTER TABLE candidates ADD COLUMN verified REAL NOT NULL DEFAULT 0.0")
                    conn.execute("ALTER TABLE candidates ADD COLUMN recent_outcomes TEXT NOT NULL DEFAULT '[]'")
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS sources ("
                        "source_id TEXT PRIMARY KEY, "
                        "agree INTEGER NOT NULL DEFAULT 0, "
                        "disagree INTEGER NOT NULL DEFAULT 0"
                        ")"
                    )
                    # Create pending table just in case it doesn't exist
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS pending ("
                        "response_id TEXT PRIMARY KEY, "
                        "shares TEXT NOT NULL, "
                        "created REAL NOT NULL"
                        ")"
                    )
                    conn.execute("PRAGMA user_version = 1")

    def _connect(self) -> sqlite3.Connection:
        # One short-lived connection per op; WAL lets readers and a writer coexist.
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    # ---- row <-> Candidate --------------------------------------------------

    def _row_to_candidate(self, row: sqlite3.Row, now: Optional[float]) -> Candidate:
        alpha, beta = row["alpha"], row["beta"]
        if now is not None and self.gamma < 1.0:
            dt = (now - row["last_updated"]) / self.decay_unit_sec
            alpha = _decay(alpha, self.gamma, dt)
            beta = _decay(beta, self.gamma, dt)
        return Candidate(
            id=row["id"],
            content=row["content"],
            metadata=json.loads(row["metadata"]),
            alpha=alpha,
            beta=beta,
            A=row["A"],
            B=row["B"],
            fooled=row["fooled"],
            verified=row["verified"],
            recent_outcomes=json.loads(row["recent_outcomes"]),
            last_updated=row["last_updated"],
        )

    # ---- CandidateStore-compatible interface --------------------------------

    def add_candidate(self, candidate: Candidate) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO candidates "
                "(id, content, metadata, alpha, beta, A, B, fooled, verified, recent_outcomes, last_updated) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    candidate.id,
                    candidate.content,
                    json.dumps(candidate.metadata),
                    candidate.alpha,
                    candidate.beta,
                    candidate.A,
                    candidate.B,
                    candidate.fooled,
                    candidate.verified,
                    json.dumps(candidate.recent_outcomes),
                    candidate.last_updated,
                ),
            )

    def get_candidate(self, candidate_id: str, now: Optional[float] = None) -> Optional[Candidate]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
        return self._row_to_candidate(row, now) if row else None

    def list_candidates(self, now: Optional[float] = None) -> List[Candidate]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM candidates").fetchall()
        return [self._row_to_candidate(r, now) for r in rows]

    def update_candidate(self, candidate: Candidate) -> None:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE candidates SET content=?, metadata=?, alpha=?, beta=?, "
                "A=?, B=?, fooled=?, verified=?, recent_outcomes=?, last_updated=? WHERE id=?",
                (
                    candidate.content,
                    json.dumps(candidate.metadata),
                    candidate.alpha,
                    candidate.beta,
                    candidate.A,
                    candidate.B,
                    candidate.fooled,
                    candidate.verified,
                    json.dumps(candidate.recent_outcomes),
                    candidate.last_updated,
                    candidate.id,
                ),
            )
            if cur.rowcount == 0:
                raise KeyError(f"Candidate with ID {candidate.id} not found in store.")

    # ---- atomic feedback increment ------------------------------------------

    def increment(
        self,
        candidate_id: str,
        d_alpha: float,
        d_beta: float,
        d_A: float,
        d_B: float,
        d_fooled: float = 0.0,
        d_verified: float = 0.0,
        recent_outcome: Optional[float] = None,
        now: Optional[float] = None,
    ) -> None:
        """
        Atomically: decay short-term counters to `now`, then add the deltas.
        Also updates fooled, verified, and appends to recent_outcomes buffer.
        """
        if now is None:
            now = time.time()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT alpha, beta, A, B, fooled, verified, recent_outcomes, last_updated FROM candidates WHERE id=?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                raise KeyError(f"Candidate with ID {candidate_id} not found in store.")
            dt = (now - row["last_updated"]) / self.decay_unit_sec
            alpha = _decay(row["alpha"], self.gamma, dt) + d_alpha
            beta = _decay(row["beta"], self.gamma, dt) + d_beta
            A = row["A"] + d_A
            B = row["B"] + d_B
            fooled = row["fooled"] + d_fooled
            verified = row["verified"] + d_verified
            outcomes = json.loads(row["recent_outcomes"])
            if recent_outcome is not None:
                outcomes.append(recent_outcome)
                if len(outcomes) > 30:  # N=30 ring buffer limit
                    outcomes.pop(0)
            conn.execute(
                "UPDATE candidates SET alpha=?, beta=?, A=?, B=?, fooled=?, verified=?, recent_outcomes=?, last_updated=? WHERE id=?",
                (alpha, beta, A, B, fooled, verified, json.dumps(outcomes), now, candidate_id),
            )
            conn.execute("COMMIT")
        finally:
            conn.close()

    # ---- pending credit shares (the retrieve <-> feedback bridge) -----------

    def save_pending(self, response_id: str, shares: Dict[str, float], now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pending (response_id, shares, created) VALUES (?,?,?)",
                (response_id, json.dumps(shares), now),
            )

    def pop_pending(self, response_id: str) -> Optional[Dict[str, float]]:
        """Atomically fetch-and-delete the frozen shares for a response."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT shares FROM pending WHERE response_id=?", (response_id,)
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                return None
            conn.execute("DELETE FROM pending WHERE response_id=?", (response_id,))
            conn.execute("COMMIT")
            return json.loads(row["shares"])
        finally:
            conn.close()

    def gc_pending(self, max_age_sec: float, now: Optional[float] = None) -> int:
        """Expire un-acted-on pending rows. Returns number deleted."""
        if now is None:
            now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM pending WHERE created < ?", (now - max_age_sec,)
            )
            return cur.rowcount
