import os
import pandas as pd
import numpy as np
from typing import Dict, Any

class TemporalAnalystAgent:
    """
    Agent 5: Temporal Analyst (Longitudinal Progression).
    Parses historical clinical visits to calculate structural degradation velocities.
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.DataFrame()
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            print(f"[+] Temporal Analyst initialized with real OASIS dataset. Total records: {len(self.df)}")
        else:
            print("[!] Warning: real oasis_longitudinal.csv not found at target path.")

    def calculate_progression_trajectory(self, subject_id: str) -> Dict[str, Any]:
        """Calculates personalized longitudinal drift metrics for a patient over time."""
        if self.df.empty:
            return {"status": "No historical database loaded.", "atrophy_velocity_pct": 0.0, "mmse_drift": 0.0}
            
        # Filter all historical visits for this specific patient, sorted by chronological order
        patient_history = self.df[self.df['Subject ID'] == subject_id].sort_values(by='Visit')
        num_visits = len(patient_history)
        
        if num_visits <= 1:
            return {
                "visits_tracked": num_visits,
                "atrophy_velocity_pct": 0.0,
                "mmse_drift": 0.0,
                "clinical_trend": "Baseline Only (Insufficient Longitudinal Points)"
            }
            
        # FIXED: Look for 'MR Delay' exactly as it is formatted in the official CSV
        days_elapsed = float(patient_history['MR Delay'].max() - patient_history['MR Delay'].min())
        years_elapsed = days_elapsed / 365.25 if days_elapsed > 0 else 1.0
        
        # 2. Calculate Atrophy Velocity: Change in Normalized Whole Brain Volume (nWBV) per year
        baseline_nwbv = float(patient_history['nWBV'].iloc[0])
        recent_nwbv = float(patient_history['nWBV'].iloc[-1])
        
        # Brain tissue loss expressed as a positive percentage per year
        total_loss = baseline_nwbv - recent_nwbv
        atrophy_velocity = (total_loss / baseline_nwbv) / years_elapsed * 100
        
        # 3. Calculate Cognitive Drift: Total change in MMSE across tracking timeline
        baseline_mmse = float(patient_history['MMSE'].iloc[0])
        recent_mmse = float(patient_history['MMSE'].iloc[-1])
        mmse_drift = recent_mmse - baseline_mmse
        
        # Determine tracking categorization
        if atrophy_velocity > 1.5 or mmse_drift <= -3:
            trend = "Aggressive Cognitive/Structural Decline"
        elif atrophy_velocity > 0.5:
            trend = "Typical Age-Related Neuro-Degradation"
        else:
            trend = "Stable / Linear Neuro-Maintenance"
            
        return {
            "visits_tracked": num_visits,
            "years_monitored": round(years_elapsed, 2),
            "atrophy_velocity_pct": round(atrophy_velocity, 3),
            "mmse_drift": int(mmse_drift),
            "clinical_trend": trend
        }

if __name__ == "__main__":
    LONG_CSV = os.path.join("data", "oasis_raw", "oasis_longitudinal.csv")
    analyst = TemporalAnalystAgent(LONG_CSV)
    
    metrics = analyst.calculate_progression_trajectory("OAS2_0001")
    print(f"\n--- Longitudinal Trajectory Analytics Output ---")
    for k, v in metrics.items():
        print(f"{k.replace('_', ' ').title()}: {v}")
        