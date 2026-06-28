"""
Persistent SQLite-backed CandidateStore (Phase 4).

Drop-in for the in-memory CandidateStore, plus support for query-conditional counters,
recency-based decay, settings persistence, and atomic increments.
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

from .store import Candidate, CandidateStore


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
    cluster_counters TEXT NOT NULL DEFAULT '{}',
    last_confirmed  REAL NOT NULL,
    last_updated    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS pending (
    response_id TEXT PRIMARY KEY,
    shares      TEXT NOT NULL,   -- JSON {candidate_id: credit_share}
    cluster_id  TEXT,            -- Query cluster
    created     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    agree     INTEGER NOT NULL DEFAULT 0,
    disagree  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS settings (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL
);
"""


def _decay(value: float, gamma: float, dt_units: float) -> float:
    """Beta-counter decay toward the prior of 1.0:  x <- 1 + (x-1) * gamma^dt."""
    if gamma >= 1.0 or dt_units <= 0:
        return value
    return 1.0 + (value - 1.0) * (gamma ** dt_units)


class SqliteCandidateStore(CandidateStore):
    def __init__(
        self,
        db_path: str,
        gamma: float = 1.0,
        decay_unit_sec: float = 86400.0,
    ):
        super().__init__()
        self.db_path = db_path
        self.gamma = gamma
        self.decay_unit_sec = decay_unit_sec
        
        with self._txn() as conn:
            # Check if candidates table exists
            table_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='candidates'"
            ).fetchone()
            
            if not table_exists:
                # Fresh DB: create tables using modern schema and set user_version=2
                conn.executescript(SCHEMA_MODERN)
                conn.execute("PRAGMA user_version = 2")
            else:
                # Existing DB: check user_version
                version_row = conn.execute("PRAGMA user_version").fetchone()
                user_version = version_row[0] if version_row else 0
                
                if user_version < 1:
                    # Run schema migrations to match version 1
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
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS pending ("
                        "response_id TEXT PRIMARY KEY, "
                        "shares TEXT NOT NULL, "
                        "created REAL NOT NULL"
                        ")"
                    )
                
                # Upgrade to Version 2: Add query clustering & settings
                if user_version < 2:
                    conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                    try:
                        conn.execute("ALTER TABLE candidates ADD COLUMN cluster_counters TEXT NOT NULL DEFAULT '{}'")
                    except sqlite3.OperationalError:
                        pass
                    try:
                        conn.execute(f"ALTER TABLE candidates ADD COLUMN last_confirmed REAL NOT NULL DEFAULT {time.time()}")
                    except sqlite3.OperationalError:
                        pass
                    try:
                        conn.execute("ALTER TABLE pending ADD COLUMN cluster_id TEXT")
                    except sqlite3.OperationalError:
                        pass
                    conn.execute("PRAGMA user_version = 2")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def _txn(self):
        """Connection context that commits on success and ALWAYS closes.

        sqlite3's native `with conn:` only commits the transaction — it does
        NOT close the connection, which leaks handles (the ResourceWarning).
        This wraps it so every short-lived op cleans up after itself.
        """
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- row <-> Candidate --------------------------------------------------

    def _row_to_candidate(self, row: sqlite3.Row, now: Optional[float]) -> Candidate:
        alpha, beta = row["alpha"], row["beta"]
        last_confirmed = row["last_confirmed"]
        if now is not None and self.gamma < 1.0:
            dt = (now - last_confirmed) / self.decay_unit_sec
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
            cluster_counters=json.loads(row["cluster_counters"]),
            last_confirmed=last_confirmed,
            last_updated=row["last_updated"],
        )

    # ---- CandidateStore-compatible interface --------------------------------

    def add_candidate(self, candidate: Candidate) -> None:
        with self._txn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO candidates "
                "(id, content, metadata, alpha, beta, A, B, fooled, verified, recent_outcomes, cluster_counters, last_confirmed, last_updated) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                    json.dumps(candidate.cluster_counters),
                    candidate.last_confirmed,
                    candidate.last_updated,
                ),
            )

    def get_candidate(self, candidate_id: str, now: Optional[float] = None) -> Optional[Candidate]:
        with self._txn() as conn:
            row = conn.execute(
                "SELECT * FROM candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
        return self._row_to_candidate(row, now) if row else None

    def list_candidates(self, now: Optional[float] = None) -> List[Candidate]:
        with self._txn() as conn:
            rows = conn.execute("SELECT * FROM candidates").fetchall()
        return [self._row_to_candidate(r, now) for r in rows]

    def update_candidate(self, candidate: Candidate) -> None:
        with self._txn() as conn:
            cur = conn.execute(
                "UPDATE candidates SET content=?, metadata=?, alpha=?, beta=?, "
                "A=?, B=?, fooled=?, verified=?, recent_outcomes=?, cluster_counters=?, last_confirmed=?, last_updated=? WHERE id=?",
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
                    json.dumps(candidate.cluster_counters),
                    candidate.last_confirmed,
                    candidate.last_updated,
                    candidate.id,
                ),
            )
            if cur.rowcount == 0:
                raise KeyError(f"Candidate with ID {candidate.id} not found in store.")

    # ---- settings persistence -----------------------------------------------

    def save_setting(self, key: str, value: str) -> None:
        with self._txn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, value),
            )

    def get_setting(self, key: str) -> Optional[str]:
        with self._txn() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

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
        cluster_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> None:
        """
        Atomically: decay short-term counters based on time since last_confirmed, then add deltas.
        If cluster_id is specified, decays and updates that cluster's counters as well.
        Updates last_confirmed to `now` if recent_outcome > 0.5 (indicating verification success).
        """
        if now is None:
            now = time.time()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT alpha, beta, A, B, fooled, verified, recent_outcomes, cluster_counters, last_confirmed, last_updated FROM candidates WHERE id=?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                raise KeyError(f"Candidate with ID {candidate_id} not found in store.")
            
            last_confirmed = row["last_confirmed"]
            dt = (now - last_confirmed) / self.decay_unit_sec
            alpha = _decay(row["alpha"], self.gamma, dt) + d_alpha
            beta = _decay(row["beta"], self.gamma, dt) + d_beta
            A = row["A"] + d_A
            B = row["B"] + d_B
            fooled = row["fooled"] + d_fooled
            verified = row["verified"] + d_verified
            outcomes = json.loads(row["recent_outcomes"])
            if recent_outcome is not None:
                outcomes.append(recent_outcome)
                if len(outcomes) > 30:
                    outcomes.pop(0)

            # Update conditional cluster counters if cluster_id is set
            cluster_counters = json.loads(row["cluster_counters"])
            if cluster_id:
                if cluster_id not in cluster_counters:
                    cluster_counters[cluster_id] = {
                        "alpha": 1.0,
                        "beta": 1.0,
                        "A": 1.0,
                        "B": 1.0,
                        "fooled": 0.0,
                        "verified": 0.0,
                        "recent_outcomes": [],
                        "last_confirmed": now,
                    }
                cc = cluster_counters[cluster_id]
                cc_lc = cc.get("last_confirmed", now)
                cc_dt = (now - cc_lc) / self.decay_unit_sec
                cc["alpha"] = _decay(cc["alpha"], self.gamma, cc_dt) + d_alpha
                cc["beta"] = _decay(cc["beta"], self.gamma, cc_dt) + d_beta
                cc["A"] = cc.get("A", 1.0) + d_A
                cc["B"] = cc.get("B", 1.0) + d_B
                cc["fooled"] = cc.get("fooled", 0.0) + d_fooled
                cc["verified"] = cc.get("verified", 0.0) + d_verified
                cc_outcomes = cc.get("recent_outcomes", [])
                if recent_outcome is not None:
                    cc_outcomes.append(recent_outcome)
                    if len(cc_outcomes) > 30:
                        cc_outcomes.pop(0)
                cc["recent_outcomes"] = cc_outcomes
                if recent_outcome is not None and recent_outcome > 0.5:
                    cc["last_confirmed"] = now
                cluster_counters[cluster_id] = cc

            # Update global last_confirmed if verification/outcome is positive
            if recent_outcome is not None and recent_outcome > 0.5:
                last_confirmed = now

            conn.execute(
                "UPDATE candidates SET alpha=?, beta=?, A=?, B=?, fooled=?, verified=?, recent_outcomes=?, cluster_counters=?, last_confirmed=?, last_updated=? WHERE id=?",
                (alpha, beta, A, B, fooled, verified, json.dumps(outcomes), json.dumps(cluster_counters), last_confirmed, now, candidate_id),
            )
            conn.execute("COMMIT")
        finally:
            conn.close()

    # ---- pending credit shares (the retrieve <-> feedback bridge) -----------

    def save_pending(self, response_id: str, shares: Dict[str, float], cluster_id: Optional[str] = None, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        with self._txn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pending (response_id, shares, cluster_id, created) VALUES (?,?,?,?)",
                (response_id, json.dumps(shares), cluster_id, now),
            )

    def pop_pending(self, response_id: str) -> Optional[Tuple[Dict[str, float], Optional[str]]]:
        """Atomically fetch-and-delete the frozen shares and cluster_id for a response."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT shares, cluster_id FROM pending WHERE response_id=?", (response_id,)
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                return None
            conn.execute("DELETE FROM pending WHERE response_id=?", (response_id,))
            conn.execute("COMMIT")
            return json.loads(row["shares"]), row["cluster_id"]
        finally:
            conn.close()

    def gc_pending(self, max_age_sec: float, now: Optional[float] = None) -> int:
        """Expire un-acted-on pending rows. Returns number deleted."""
        if now is None:
            now = time.time()
        with self._txn() as conn:
            conn.execute(
                "DELETE FROM pending WHERE created < ?", (now - max_age_sec,)
            )
            return conn.total_changes
