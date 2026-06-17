import os
import sys
import numpy as np

# 1. Dynamically calculate absolute path locations
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # src/orchestrator
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))  # src
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, '..')) # Project Root

# 2. Force-inject both paths right to the front of Python's lookup list
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 3. Clean, direct imports (dropped the temperamental 'src.' prefix)
from orchestrator.chief_medical_officer import AdvancedChiefMedicalOfficer
from orchestrator.hitl_queue import ActiveLearningRegistry
from pipeline.onnx_inference.onnx_agent import ONNXMultimodalFusionAgent

def run_advanced_terminal_pipeline():
    print("=======================================================")
    print("  [SYSTEM BOOT] Advanced Multi-Agent Swarm Connected   ")
    print("=======================================================")
    
    # Boot up core multi-agent framework
    cmo = AdvancedChiefMedicalOfficer(workspace_root=PROJECT_ROOT)
    
    # Boot up our newly quantized 12.11 MB hardware asset
    quant_model_path = os.path.join(PROJECT_ROOT, "src", "pipeline", "onnx_inference", "multimodal_fusion_int8.onnx")
    onnx_engine = ONNXMultimodalFusionAgent(quant_model_path)
    
    # Connect our SQLite database registry
    db_path = os.path.join(PROJECT_ROOT, "data", "active_learning.db")
    registry = ActiveLearningRegistry(db_path)
    print("=======================================================\n")

    # Pull a real patient row from the dataset (Index 0)
    patient_row = cmo.patient_df.iloc[0]
    subject_id = str(patient_row['Subject_ID'])
    print(f"[+] Processing Patient Profile: {subject_id}")
    
    # Fetch chronological tracking metrics from Agent 5
    long_metrics = cmo.temporal_agent.calculate_progression_trajectory("OAS2_0001")
    atrophy_vel = float(long_metrics.get('atrophy_velocity_pct', 0.0))
    
    # Gather mock raw inputs for a quick forward pass verification
    mock_mri = np.random.randn(1, 1, 224, 224).astype(np.float32)
    
    # STRESS-TEST SCENARIO: We force a severe MMSE drop (5.0) to trigger Agent 6
    simulated_mmse = 5.0
    biomarker_array = np.array([float(patient_row['Age']), simulated_mmse, atrophy_vel, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Compute hardware-native inference on your Snapdragon-optimized graph
    print("[*] Dispatching inputs to Quantized Cross-Attention ONNX Engine...")
    pred_class, confidence = onnx_engine.execute_hardware_inference(mock_mri, biomarker_array)
    print(f" -> Engine Diagnostic Output: {pred_class} ({confidence:.2f}% confidence)")

    # Run the Ethicist Guardrail Audit (Agent 6)
    print("[*] Running Ethicist Agent safety verification check...")
    is_flagged, reason = cmo.ethicist_agent.audit_diagnostic_proposal(
        predicted_class=pred_class,
        confidence=confidence,
        mmse_score=simulated_mmse,
        atrophy_velocity=atrophy_vel
    )

    if is_flagged:
        print("\n[🚨 CRITICAL ALARM] Ethicist intercepted prediction!")
        print(f" -> Reason: {reason}")
        print("[*] Serializing case logs directly into active_learning.db SQLite layer...")
        
        # Write directly to the relational database tables
        registry.log_flagged_anomaly(
            patient_id=subject_id,
            age=float(patient_row['Age']),
            mmse=simulated_mmse,
            velocity=atrophy_vel,
            img_path="data/oasis_raw/Non Demented/OAS1_0001_MR1_mpr-1_100.cef",
            pred=pred_class,
            conf=confidence,
            reason=reason
        )
        
        # Verify the data was written by reading back the active queue
        queue = registry.fetch_active_queue()
        print(f"[SUCCESS] Database updated! Active cases in the human review queue: {len(queue)}")
    else:
        print("\n[✔ COMPLIANCE PASSED] Diagnosis cleared for hospital charts.")

if __name__ == "__main__":
    run_advanced_terminal_pipeline()