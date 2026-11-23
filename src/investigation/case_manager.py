
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


class CaseManager:

    def __init__(self, db_path="data/fraudlens_cases.db"):

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_database()

    def _connect(self):

        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self):

        with self._connect() as conn:

            conn.execute("""
                CREATE TABLE IF NOT EXISTS investigation_cases (
                    case_id TEXT PRIMARY KEY,
                    transaction_id TEXT UNIQUE NOT NULL,
                    transaction_data TEXT NOT NULL,
                    fraud_probability REAL NOT NULL,
                    risk_score INTEGER NOT NULL,
                    risk_tier TEXT NOT NULL,
                    review_flag INTEGER NOT NULL,
                    recommended_action TEXT,
                    status TEXT NOT NULL DEFAULT 'Pending',
                    analyst_notes TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS case_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT DEFAULT '',
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (case_id)
                        REFERENCES investigation_cases(case_id)
                )
            """)

    @staticmethod
    def _now():

        return datetime.now(timezone.utc).isoformat()

    def import_transactions(
        self,
        csv_path="reports/risk/transaction_risk_scores.csv"
    ):

        path = Path(csv_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Risk report not found: {path}"
            )

        df = pd.read_csv(path)

        required_columns = [
            "fraud_probability",
            "risk_score",
            "risk_tier",
            "review_flag",
            "recommended_action"
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        imported = 0

        with self._connect() as conn:

            for index, row in df.iterrows():

                transaction_id = f"TX-{index + 1:06d}"
                case_id = f"FL-{index + 1:06d}"

                transaction_data = row.to_dict()

                # Keep actual labels out of the analyst-facing
                # transaction payload.
                transaction_data.pop("actual_fraud", None)
                transaction_data.pop("fraud", None)

                transaction_json = json.dumps(
                    transaction_data,
                    default=str
                )

                now = self._now()

                cursor = conn.execute("""
                    INSERT OR IGNORE INTO investigation_cases (
                        case_id,
                        transaction_id,
                        transaction_data,
                        fraud_probability,
                        risk_score,
                        risk_tier,
                        review_flag,
                        recommended_action,
                        status,
                        analyst_notes,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Pending', '', ?, ?)
                """, (
                    case_id,
                    transaction_id,
                    transaction_json,
                    float(row["fraud_probability"]),
                    int(row["risk_score"]),
                    str(row["risk_tier"]),
                    int(row["review_flag"]),
                    str(row["recommended_action"]),
                    now,
                    now
                ))

                if cursor.rowcount > 0:
                    imported += 1

        return imported

    def get_cases(
        self,
        status=None,
        risk_tier=None,
        flagged_only=False
    ):

        query = """
            SELECT *
            FROM investigation_cases
            WHERE 1 = 1
        """

        params = []

        if status and status != "All":
            query += " AND status = ?"
            params.append(status)

        if risk_tier and risk_tier != "All":
            query += " AND risk_tier = ?"
            params.append(risk_tier)

        if flagged_only:
            query += " AND review_flag = 1"

        query += """
            ORDER BY risk_score DESC, created_at DESC
        """

        with self._connect() as conn:

            rows = conn.execute(
                query,
                params
            ).fetchall()

        records = []

        for row in rows:

            item = dict(row)

            transaction_data = json.loads(
                item.pop("transaction_data")
            )

            item.update(transaction_data)

            records.append(item)

        return pd.DataFrame(records)

    def get_case(self, case_id):

        with self._connect() as conn:

            row = conn.execute("""
                SELECT *
                FROM investigation_cases
                WHERE case_id = ?
            """, (case_id,)).fetchone()

        if row is None:
            return None

        result = dict(row)

        result["transaction_data"] = json.loads(
            result["transaction_data"]
        )

        return result

    def update_status(
        self,
        case_id,
        new_status,
        details=""
    ):

        valid_statuses = [
            "Pending",
            "Investigating",
            "Confirmed Fraud",
            "Legitimate"
        ]

        if new_status not in valid_statuses:
            raise ValueError(
                f"Invalid status: {new_status}"
            )

        now = self._now()

        with self._connect() as conn:

            cursor = conn.execute("""
                UPDATE investigation_cases
                SET status = ?, updated_at = ?
                WHERE case_id = ?
            """, (
                new_status,
                now,
                case_id
            ))

            if cursor.rowcount == 0:
                raise ValueError(
                    f"Case not found: {case_id}"
                )

            conn.execute("""
                INSERT INTO case_history (
                    case_id,
                    action,
                    details,
                    timestamp
                )
                VALUES (?, ?, ?, ?)
            """, (
                case_id,
                f"Status changed to {new_status}",
                details,
                now
            ))

    def add_note(self, case_id, note):

        note = note.strip()

        if not note:
            raise ValueError("Note cannot be empty.")

        now = self._now()

        with self._connect() as conn:

            row = conn.execute("""
                SELECT analyst_notes
                FROM investigation_cases
                WHERE case_id = ?
            """, (case_id,)).fetchone()

            if row is None:
                raise ValueError(
                    f"Case not found: {case_id}"
                )

            existing_notes = row["analyst_notes"] or ""

            timestamped_note = (
                f"[{now}] {note}"
            )

            updated_notes = (
                existing_notes + "\n" + timestamped_note
                if existing_notes
                else timestamped_note
            )

            conn.execute("""
                UPDATE investigation_cases
                SET analyst_notes = ?, updated_at = ?
                WHERE case_id = ?
            """, (
                updated_notes,
                now,
                case_id
            ))

            conn.execute("""
                INSERT INTO case_history (
                    case_id,
                    action,
                    details,
                    timestamp
                )
                VALUES (?, 'Note added', ?, ?)
            """, (
                case_id,
                note,
                now
            ))

    def get_history(self, case_id):

        with self._connect() as conn:

            rows = conn.execute("""
                SELECT action, details, timestamp
                FROM case_history
                WHERE case_id = ?
                ORDER BY history_id DESC
            """, (case_id,)).fetchall()

        return pd.DataFrame(
            [dict(row) for row in rows]
        )

    def get_summary(self):

        with self._connect() as conn:

            total = conn.execute("""
                SELECT COUNT(*) FROM investigation_cases
            """).fetchone()[0]

            flagged = conn.execute("""
                SELECT COUNT(*)
                FROM investigation_cases
                WHERE review_flag = 1
            """).fetchone()[0]

            statuses = conn.execute("""
                SELECT status, COUNT(*) AS count
                FROM investigation_cases
                GROUP BY status
            """).fetchall()

            tiers = conn.execute("""
                SELECT risk_tier, COUNT(*) AS count
                FROM investigation_cases
                GROUP BY risk_tier
            """).fetchall()

        return {
            "total_cases": total,
            "flagged_cases": flagged,
            "statuses": {
                row["status"]: row["count"]
                for row in statuses
            },
            "risk_distribution": {
                row["risk_tier"]: row["count"]
                for row in tiers
            }
        }
