import sqlite3
import os
import json
import time
from typing import Optional, Dict, Any, List

LEDGER_PATH = "/DATA/horizon_disentangled/manifests/ingestion_ledger.sqlite"

class IngestionLedger:
    def __init__(self, db_path: str = LEDGER_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_ledger (
                patient_id TEXT PRIMARY KEY,
                cohort TEXT NOT NULL,
                domain TEXT NOT NULL,
                status TEXT NOT NULL,
                shard_file TEXT,
                num_visits INTEGER DEFAULT 0,
                raw_size_bytes INTEGER DEFAULT 0,
                hdf5_size_bytes INTEGER DEFAULT 0,
                checksums TEXT,
                error_msg TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON ingestion_ledger(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cohort ON ingestion_ledger(cohort)")

    def register_patient(self, patient_id: str, cohort: str, domain: str, raw_size_bytes: int = 0):
        now = time.time()
        with self._get_conn() as conn:
            conn.execute("""
            INSERT INTO ingestion_ledger (patient_id, cohort, domain, status, raw_size_bytes, created_at, updated_at)
            VALUES (?, ?, ?, 'PENDING', ?, ?, ?)
            ON CONFLICT(patient_id) DO UPDATE SET
                updated_at = excluded.updated_at
            """, (patient_id, cohort, domain, raw_size_bytes, now, now))

    def update_status(self, patient_id: str, status: str, shard_file: Optional[str] = None,
                      num_visits: Optional[int] = None, raw_size_bytes: Optional[int] = None,
                      hdf5_size_bytes: Optional[int] = None,
                      checksums: Optional[Dict[str, str]] = None, error_msg: Optional[str] = None):
        now = time.time()
        updates = ["status = ?", "updated_at = ?"]
        params: List[Any] = [status, now]

        if shard_file is not None:
            updates.append("shard_file = ?")
            params.append(shard_file)
        if num_visits is not None:
            updates.append("num_visits = ?")
            params.append(num_visits)
        if raw_size_bytes is not None:
            updates.append("raw_size_bytes = ?")
            params.append(raw_size_bytes)
        if hdf5_size_bytes is not None:
            updates.append("hdf5_size_bytes = ?")
            params.append(hdf5_size_bytes)
        if checksums is not None:
            updates.append("checksums = ?")
            params.append(json.dumps(checksums))
        if error_msg is not None:
            updates.append("error_msg = ?")
            params.append(error_msg)

        params.append(patient_id)
        sql = f"UPDATE ingestion_ledger SET {', '.join(updates)} WHERE patient_id = ?"
        with self._get_conn() as conn:
            conn.execute(sql, params)

    def get_status(self, patient_id: str) -> Optional[str]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT status FROM ingestion_ledger WHERE patient_id = ?", (patient_id,)).fetchone()
            return row["status"] if row else None

    def get_all_by_cohort(self, cohort: str) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM ingestion_ledger WHERE cohort = ?", (cohort,)).fetchall()
            return [dict(r) for r in rows]

    def get_summary(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM ingestion_ledger").fetchone()["c"]
            by_status = conn.execute("SELECT status, COUNT(*) as c FROM ingestion_ledger GROUP BY status").fetchall()
            by_cohort = conn.execute("SELECT cohort, status, COUNT(*) as c FROM ingestion_ledger GROUP BY cohort, status").fetchall()
            return {
                "total_patients": total,
                "by_status": {r["status"]: r["c"] for r in by_status},
                "by_cohort": [{ "cohort": r["cohort"], "status": r["status"], "count": r["c"] } for r in by_cohort]
            }
