import sqlite3
import os
from typing import List, Dict, Any
from datetime import datetime, timezone


class ActiveLearningRegistry:
    """
    Human-in-the-Loop (HITL) Queue Manager.
    Maintains a SQLite database of flagged cases requiring human review.
    Supports active learning workflows for continuous model improvement.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize database schema
        self._initialize_database()

    def _initialize_database(self):
        """Creates the database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Table for flagged anomalies requiring human review
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flagged_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                age REAL,
                mmse REAL,
                atrophy_velocity REAL,
                image_path TEXT,
                predicted_class TEXT,
                confidence REAL,
                flag_reason TEXT,
                reviewed BOOLEAN DEFAULT 0,
                human_label TEXT,
                review_timestamp TEXT,
                reviewer_notes TEXT
            )
        """)

        # Table for tracking model performance metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                model_version TEXT,
                notes TEXT
            )
        """)

        conn.commit()
        conn.close()

    def log_flagged_anomaly(
        self,
        patient_id: str,
        age: float,
        mmse: float,
        velocity: float,
        img_path: str,
        pred: str,
        conf: float,
        reason: str,
    ) -> int:
        """
        Logs a flagged case to the database for human review.

        Returns:
            int: The ID of the inserted record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now(timezone.utc).isoformat()

        cursor.execute(
            """
            INSERT INTO flagged_cases 
            (patient_id, timestamp, age, mmse, atrophy_velocity, image_path, 
             predicted_class, confidence, flag_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (patient_id, timestamp, age, mmse, velocity, img_path, pred, conf, reason),
        )

        record_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return record_id

    def fetch_active_queue(self) -> List[Dict[str, Any]]:
        """
        Retrieves all unreviewed flagged cases.

        Returns:
            List of dictionaries containing case information
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM flagged_cases 
            WHERE reviewed = 0 
            ORDER BY timestamp DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def mark_reviewed(self, case_id: int, human_label: str, reviewer_notes: str = ""):
        """
        Marks a case as reviewed with human expert label.

        Args:
            case_id: Database ID of the case
            human_label: Expert-provided correct label
            reviewer_notes: Optional notes from the reviewer
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        review_timestamp = datetime.now(timezone.utc).isoformat()

        cursor.execute(
            """
            UPDATE flagged_cases 
            SET reviewed = 1,
                human_label = ?,
                review_timestamp = ?,
                reviewer_notes = ?
            WHERE id = ?
        """,
            (human_label, review_timestamp, reviewer_notes, case_id),
        )

        conn.commit()
        conn.close()

    def log_performance_metric(
        self, metric_name: str, metric_value: float, model_version: str = "1.0", notes: str = ""
    ):
        """
        Logs a performance metric for tracking model improvements.

        Args:
            metric_name: Name of the metric (e.g., "accuracy", "f1_score")
            metric_value: Numerical value of the metric
            model_version: Version identifier of the model
            notes: Optional notes about the metric
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now(timezone.utc).isoformat()

        cursor.execute(
            """
            INSERT INTO performance_metrics 
            (timestamp, metric_name, metric_value, model_version, notes)
            VALUES (?, ?, ?, ?, ?)
        """,
            (timestamp, metric_name, metric_value, model_version, notes),
        )

        conn.commit()
        conn.close()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Returns summary statistics about the HITL queue.

        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total flagged cases
        cursor.execute("SELECT COUNT(*) FROM flagged_cases")
        total_flagged = cursor.fetchone()[0]

        # Unreviewed cases
        cursor.execute("SELECT COUNT(*) FROM flagged_cases WHERE reviewed = 0")
        unreviewed = cursor.fetchone()[0]

        # Reviewed cases
        cursor.execute("SELECT COUNT(*) FROM flagged_cases WHERE reviewed = 1")
        reviewed = cursor.fetchone()[0]

        # Average confidence of flagged cases
        cursor.execute("SELECT AVG(confidence) FROM flagged_cases")
        avg_confidence = cursor.fetchone()[0] or 0.0

        conn.close()

        return {
            "total_flagged": total_flagged,
            "unreviewed": unreviewed,
            "reviewed": reviewed,
            "review_rate": (reviewed / total_flagged * 100) if total_flagged > 0 else 0.0,
            "avg_confidence": avg_confidence,
        }

    def export_training_data(self, output_path: str):
        """
        Exports reviewed cases for model retraining.

        Args:
            output_path: Path to save the CSV file
        """
        import pandas as pd

        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT patient_id, age, mmse, atrophy_velocity, 
                   image_path, human_label, reviewer_notes
            FROM flagged_cases 
            WHERE reviewed = 1 AND human_label IS NOT NULL
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        df.to_csv(output_path, index=False)
        print(f"[+] Exported {len(df)} reviewed cases to {output_path}")


if __name__ == "__main__":
    # Test the registry
    test_db = "test_active_learning.db"
    registry = ActiveLearningRegistry(test_db)

    # Log a test case
    case_id = registry.log_flagged_anomaly(
        patient_id="TEST_001",
        age=75.0,
        mmse=12.0,
        velocity=2.5,
        img_path="test/image.jpg",
        pred="Moderate Dementia",
        conf=45.0,
        reason="Low confidence score",
    )

    print(f"[+] Logged test case with ID: {case_id}")

    # Fetch active queue
    queue = registry.fetch_active_queue()
    print(f"[+] Active queue size: {len(queue)}")

    # Get statistics
    stats = registry.get_statistics()
    print(f"[+] Statistics: {stats}")

    # Clean up test database
    if os.path.exists(test_db):
        os.remove(test_db)
        print("[+] Cleaned up test database")
