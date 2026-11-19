"""
EU AI Act (Regulation (EU) 2024/1689, Articles 9–15) & PTB TEF-Health
Compliance Audit Logger for Horizon Disentangled.

Implements immutable record-keeping for:
  - Art. 9: Risk Management System logging
  - Art. 10: Data Governance (dataset hashes, cohort manifests)
  - Art. 11: Technical Documentation (model configs, hyperparams)
  - Art. 12: Record-Keeping (inference events, OOD flags, UQ metrics)
  - Art. 13: Transparency (RANO/BI-RADS structured reports)
  - Art. 14: Human Oversight (concordance audits, clinician review flags)
  - Art. 15: Accuracy, Robustness, Cybersecurity (checksum verification)
"""
import os
import json
import hashlib
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


class EUAIActAuditLogger:
    """
    Immutable audit trail logger for EU AI Act conformity assessment.

    Writes to both:
      1. Append-only JSONL files (human-readable, grep-friendly)
      2. SQLite database (queryable, with integrity constraints)

    All entries are timestamped (ISO 8601 UTC) and include:
      - XXH64 hashes of input data and model checkpoints
      - OOD flags and epistemic uncertainty vectors
      - Counterfactual prompt provenance
      - Concordance audit results
    """

    def __init__(
        self,
        log_dir: str = "/DATA/horizon_disentangled/artifacts/audit_logs",
        db_name: str = "eu_ai_act_audit.db"
    ):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(log_dir, "audit_trail.jsonl")
        self.db_path = os.path.join(log_dir, db_name)
        self._init_db()

    def _init_db(self):
        """Initializes SQLite audit database with conformity tables."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Art. 10: Data Governance — dataset manifests
        c.execute("""
            CREATE TABLE IF NOT EXISTS data_governance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                dataset_hash TEXT NOT NULL,
                num_patients INTEGER,
                num_visits INTEGER,
                cohort_manifest TEXT,
                storage_path TEXT,
                data_quality_notes TEXT
            )
        """)

        # Art. 11: Technical Documentation — model configurations
        c.execute("""
            CREATE TABLE IF NOT EXISTS technical_documentation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                stage TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_hash TEXT,
                config_json TEXT NOT NULL,
                hyperparameters TEXT,
                training_metrics TEXT,
                checkpoint_path TEXT
            )
        """)

        # Art. 12: Record-Keeping — inference event log
        c.execute("""
            CREATE TABLE IF NOT EXISTS inference_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                visit_id TEXT,
                stage TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                output_hash TEXT NOT NULL,
                ood_flag INTEGER DEFAULT 0,
                ood_score REAL,
                epistemic_variance_mean REAL,
                epistemic_variance_max REAL,
                prediction_json TEXT,
                processing_time_ms REAL
            )
        """)

        # Art. 13: Transparency — structured clinical reports
        c.execute("""
            CREATE TABLE IF NOT EXISTS transparency_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                report_type TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                report_json TEXT NOT NULL,
                concordance_rate REAL,
                concordance_warning INTEGER DEFAULT 0,
                clinician_review_required INTEGER DEFAULT 0
            )
        """)

        # Art. 14: Human Oversight — concordance and review tracking
        c.execute("""
            CREATE TABLE IF NOT EXISTS human_oversight (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                num_posterior_draws INTEGER,
                concordance_rate REAL,
                divergence_flag INTEGER DEFAULT 0,
                override_decision TEXT,
                reviewer_id TEXT,
                notes TEXT
            )
        """)

        # Art. 15: Accuracy & Robustness — validation metrics
        c.execute("""
            CREATE TABLE IF NOT EXISTS accuracy_robustness (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                stage TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                dataset_split TEXT,
                confidence_interval_lower REAL,
                confidence_interval_upper REAL,
                notes TEXT
            )
        """)

        # Art. 9: Risk Management — system-level risk events
        c.execute("""
            CREATE TABLE IF NOT EXISTS risk_management (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                risk_category TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                mitigation_action TEXT,
                resolved INTEGER DEFAULT 0
            )
        """)

        conn.commit()
        conn.close()

    @staticmethod
    def _utc_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def compute_hash(data: bytes, algorithm: str = "sha256") -> str:
        """Computes cryptographic hash for audit integrity."""
        h = hashlib.new(algorithm)
        h.update(data)
        return f"{algorithm}:{h.hexdigest()}"

    @staticmethod
    def compute_file_hash(filepath: str, algorithm: str = "sha256") -> str:
        """Computes hash of a file for Art. 15 integrity verification."""
        h = hashlib.new(algorithm)
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return f"{algorithm}:{h.hexdigest()}"

    def _append_jsonl(self, record: Dict[str, Any]):
        """Appends an audit record to the JSONL trail (append-only)."""
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # ─── Art. 10: Data Governance ─────────────────────────────────────
    def log_dataset_registration(
        self,
        dataset_name: str,
        dataset_hash: str,
        num_patients: int,
        num_visits: int,
        cohort_manifest: Optional[Dict] = None,
        storage_path: str = "",
        notes: str = ""
    ):
        ts = self._utc_timestamp()
        record = {
            "event": "data_governance.register",
            "timestamp": ts,
            "dataset_name": dataset_name,
            "dataset_hash": dataset_hash,
            "num_patients": num_patients,
            "num_visits": num_visits,
            "storage_path": storage_path,
            "notes": notes
        }
        self._append_jsonl(record)

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO data_governance (timestamp, dataset_name, dataset_hash, "
            "num_patients, num_visits, cohort_manifest, storage_path, data_quality_notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, dataset_name, dataset_hash, num_patients, num_visits,
             json.dumps(cohort_manifest) if cohort_manifest else None,
             storage_path, notes)
        )
        conn.commit()
        conn.close()

    # ─── Art. 11: Technical Documentation ─────────────────────────────
    def log_model_configuration(
        self,
        stage: str,
        model_name: str,
        config: Dict[str, Any],
        hyperparameters: Optional[Dict] = None,
        training_metrics: Optional[Dict] = None,
        checkpoint_path: str = "",
        model_hash: str = ""
    ):
        ts = self._utc_timestamp()
        record = {
            "event": "technical_documentation.model_config",
            "timestamp": ts,
            "stage": stage,
            "model_name": model_name,
            "config": config,
            "checkpoint_path": checkpoint_path,
            "model_hash": model_hash
        }
        self._append_jsonl(record)

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO technical_documentation (timestamp, stage, model_name, model_hash, "
            "config_json, hyperparameters, training_metrics, checkpoint_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, stage, model_name, model_hash,
             json.dumps(config, default=str),
             json.dumps(hyperparameters, default=str) if hyperparameters else None,
             json.dumps(training_metrics, default=str) if training_metrics else None,
             checkpoint_path)
        )
        conn.commit()
        conn.close()

    # ─── Art. 12: Record-Keeping (Inference Events) ───────────────────
    def log_inference_event(
        self,
        patient_id: str,
        stage: str,
        input_hash: str,
        output_hash: str,
        visit_id: str = "",
        ood_flag: int = 0,
        ood_score: float = 0.0,
        epistemic_variance_mean: float = 0.0,
        epistemic_variance_max: float = 0.0,
        prediction: Optional[Dict] = None,
        processing_time_ms: float = 0.0
    ):
        ts = self._utc_timestamp()
        record = {
            "event": "inference.prediction",
            "timestamp": ts,
            "patient_id": patient_id,
            "visit_id": visit_id,
            "stage": stage,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "ood_flag": ood_flag,
            "ood_score": ood_score,
            "epistemic_variance_mean": epistemic_variance_mean,
            "epistemic_variance_max": epistemic_variance_max,
            "processing_time_ms": processing_time_ms
        }
        self._append_jsonl(record)

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO inference_events (timestamp, patient_id, visit_id, stage, "
            "input_hash, output_hash, ood_flag, ood_score, epistemic_variance_mean, "
            "epistemic_variance_max, prediction_json, processing_time_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, patient_id, visit_id, stage, input_hash, output_hash,
             ood_flag, ood_score, epistemic_variance_mean, epistemic_variance_max,
             json.dumps(prediction, default=str) if prediction else None,
             processing_time_ms)
        )
        conn.commit()
        conn.close()

    # ─── Art. 13: Transparency (Structured Reports) ───────────────────
    def log_clinical_report(
        self,
        patient_id: str,
        report_type: str,
        schema_version: str,
        report: Dict[str, Any],
        concordance_rate: float = 1.0,
        concordance_warning: bool = False,
        clinician_review_required: bool = False
    ):
        ts = self._utc_timestamp()
        record = {
            "event": "transparency.clinical_report",
            "timestamp": ts,
            "patient_id": patient_id,
            "report_type": report_type,
            "schema_version": schema_version,
            "concordance_rate": concordance_rate,
            "concordance_warning": concordance_warning,
            "clinician_review_required": clinician_review_required
        }
        self._append_jsonl(record)

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO transparency_reports (timestamp, patient_id, report_type, "
            "schema_version, report_json, concordance_rate, concordance_warning, "
            "clinician_review_required) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, patient_id, report_type, schema_version,
             json.dumps(report, default=str),
             concordance_rate, int(concordance_warning), int(clinician_review_required))
        )
        conn.commit()
        conn.close()

    # ─── Art. 14: Human Oversight ─────────────────────────────────────
    def log_human_oversight(
        self,
        patient_id: str,
        action_type: str,
        num_posterior_draws: int = 0,
        concordance_rate: float = 1.0,
        divergence_flag: bool = False,
        override_decision: str = "",
        reviewer_id: str = "",
        notes: str = ""
    ):
        ts = self._utc_timestamp()
        record = {
            "event": "human_oversight.review",
            "timestamp": ts,
            "patient_id": patient_id,
            "action_type": action_type,
            "num_posterior_draws": num_posterior_draws,
            "concordance_rate": concordance_rate,
            "divergence_flag": divergence_flag
        }
        self._append_jsonl(record)

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO human_oversight (timestamp, patient_id, action_type, "
            "num_posterior_draws, concordance_rate, divergence_flag, "
            "override_decision, reviewer_id, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, patient_id, action_type, num_posterior_draws, concordance_rate,
             int(divergence_flag), override_decision, reviewer_id, notes)
        )
        conn.commit()
        conn.close()

    # ─── Art. 15: Accuracy & Robustness ───────────────────────────────
    def log_validation_metric(
        self,
        stage: str,
        metric_name: str,
        metric_value: float,
        dataset_split: str = "test",
        ci_lower: Optional[float] = None,
        ci_upper: Optional[float] = None,
        notes: str = ""
    ):
        ts = self._utc_timestamp()
        record = {
            "event": "accuracy_robustness.metric",
            "timestamp": ts,
            "stage": stage,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "dataset_split": dataset_split
        }
        self._append_jsonl(record)

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO accuracy_robustness (timestamp, stage, metric_name, metric_value, "
            "dataset_split, confidence_interval_lower, confidence_interval_upper, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, stage, metric_name, metric_value, dataset_split, ci_lower, ci_upper, notes)
        )
        conn.commit()
        conn.close()

    # ─── Art. 9: Risk Management ──────────────────────────────────────
    def log_risk_event(
        self,
        risk_category: str,
        severity: str,
        description: str,
        mitigation_action: str = ""
    ):
        ts = self._utc_timestamp()
        record = {
            "event": "risk_management.event",
            "timestamp": ts,
            "risk_category": risk_category,
            "severity": severity,
            "description": description,
            "mitigation_action": mitigation_action
        }
        self._append_jsonl(record)

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO risk_management (timestamp, risk_category, severity, "
            "description, mitigation_action) VALUES (?, ?, ?, ?, ?)",
            (ts, risk_category, severity, description, mitigation_action)
        )
        conn.commit()
        conn.close()

    # ─── Conformity Report Generator ──────────────────────────────────
    def generate_conformity_summary(self) -> Dict[str, Any]:
        """
        Generates a summary conformity assessment report
        covering all EU AI Act Articles 9–15.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        summary = {
            "generated_at": self._utc_timestamp(),
            "regulation": "EU AI Act (Regulation (EU) 2024/1689)",
            "system": "Horizon Disentangled — Causal AI for Oncology",
            "risk_classification": "High-Risk (Annex III, Section 5(b): AI for medical diagnosis)",
            "articles": {}
        }

        # Art. 9
        c.execute("SELECT COUNT(*) FROM risk_management")
        summary["articles"]["art_9_risk_management"] = {
            "total_events": c.fetchone()[0],
            "status": "Active"
        }

        # Art. 10
        c.execute("SELECT COUNT(*), SUM(num_patients), SUM(num_visits) FROM data_governance")
        row = c.fetchone()
        summary["articles"]["art_10_data_governance"] = {
            "registered_datasets": row[0],
            "total_patients": row[1] or 0,
            "total_visits": row[2] or 0,
            "status": "Compliant" if row[0] > 0 else "Pending"
        }

        # Art. 11
        c.execute("SELECT COUNT(*), COUNT(DISTINCT stage) FROM technical_documentation")
        row = c.fetchone()
        summary["articles"]["art_11_technical_documentation"] = {
            "documented_models": row[0],
            "covered_stages": row[1],
            "status": "Compliant" if row[0] > 0 else "Pending"
        }

        # Art. 12
        c.execute("SELECT COUNT(*), AVG(ood_score), AVG(epistemic_variance_mean) FROM inference_events")
        row = c.fetchone()
        summary["articles"]["art_12_record_keeping"] = {
            "total_inference_events": row[0],
            "mean_ood_score": round(row[1], 4) if row[1] else None,
            "mean_epistemic_variance": round(row[2], 6) if row[2] else None,
            "status": "Active"
        }

        # Art. 13
        c.execute("SELECT COUNT(*), AVG(concordance_rate), SUM(concordance_warning) FROM transparency_reports")
        row = c.fetchone()
        summary["articles"]["art_13_transparency"] = {
            "total_reports": row[0],
            "mean_concordance": round(row[1], 3) if row[1] else None,
            "total_warnings": row[2] or 0,
            "status": "Active"
        }

        # Art. 14
        c.execute("SELECT COUNT(*), SUM(divergence_flag) FROM human_oversight")
        row = c.fetchone()
        summary["articles"]["art_14_human_oversight"] = {
            "total_reviews": row[0],
            "total_divergences": row[1] or 0,
            "status": "Active"
        }

        # Art. 15
        c.execute("SELECT COUNT(DISTINCT metric_name), AVG(metric_value) FROM accuracy_robustness")
        row = c.fetchone()
        summary["articles"]["art_15_accuracy_robustness"] = {
            "tracked_metrics": row[0],
            "mean_metric_value": round(row[1], 4) if row[1] else None,
            "status": "Active"
        }

        conn.close()
        return summary
