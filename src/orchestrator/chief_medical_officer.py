import os
import sys
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image

# Synchronize directory structure across application layers
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(src_dir)

from agents.vision.vision_agent import AlzheimerVisionAgent
from agents.vision.explainer_agent import RadiomicsExplainerAgent
from agents.biomarker.biomarker_agent import ClinicalBiomarkerAgent
from agents.biomarker.temporal_analyst import TemporalAnalystAgent
from agents.rag.rag_agent import MedicalLibrarianAgent
from orchestrator.ethicist_agent import MedicalEthicistAgent

class AdvancedChiefMedicalOfficer:
    """
    Advanced Heterogeneous Swarm Orchestrator.
    Directs 6 independent clinical agents spanning spatial imaging,
    longitudinal kinetics, text RAG, and compliance guardrails.
    """
    def __init__(self, workspace_root: str):
        print("\n=======================================================")
        print("  [SYSTEM BOOT] Advanced Multi-Agent Swarm Connected  ")
        print("=======================================================\n")
        
        self.workspace_root = workspace_root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Establish accurate data file maps
        self.cross_csv = os.path.join(self.workspace_root, "data", "oasis_raw", "oasis_clinical_data.csv")
        self.long_csv = os.path.join(self.workspace_root, "data", "oasis_raw", "oasis_longitudinal.csv")
        
        # 1. Initialize Baseline Tabular Pipelines
        self.biomarker_agent = ClinicalBiomarkerAgent()
        self.patient_tensors, self.patient_df = self.biomarker_agent.ingest_and_process(self.cross_csv)
        
        # 2. Initialize Agent 5: Temporal Analyst
        self.temporal_agent = TemporalAnalystAgent(self.long_csv)
        
        # 3. Initialize Agent 3: Medical Librarian RAG
        self.rag_agent = MedicalLibrarianAgent()
        self.rag_agent.ingest_medical_guidelines([
            "Normal Cognition presents with an MMSE from 26 to 30. Structural brain mass is well maintained.",
            "Very Mild Dementia features slight memory changes, with an MMSE score typically hanging between 24 and 26.",
            "Mild Dementia shows progressive confusion, tracking with an MMSE score down between 20 and 24.",
            "Moderate Dementia presents severe memory degradation, tracking with an MMSE score between 13 and 20."
        ])
        
        # 4. Initialize Agent 1: Vision Architecture
        self.class_names = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
        self.vision_agent = AlzheimerVisionAgent(num_classes=len(self.class_names)).to(self.device)
        
        weights_path = os.path.join(self.workspace_root, "src", "pipeline", "onnx_inference", "best_vision_agent.pth")
        if os.path.exists(weights_path):
            self.vision_agent.load_state_dict(torch.load(weights_path, map_location=self.device))
            print("[+] Vision Agent loaded with optimized training weights.")
        else:
            print("[!] Vision weights training in separate window. Initializing untrained framework for routing tests.")
            
        # 5. Initialize Agent 4: Radiomics Explainer (Grad-CAM)
        self.explainer_agent = RadiomicsExplainerAgent(self.vision_agent)
        
        # 6. Initialize Agent 6: Medical Ethicist Guardrail
        self.ethicist_agent = MedicalEthicistAgent(confidence_floor=60.0)
        
        self.image_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        print("\n[SUCCESS] Hexagonal Multi-Agent Architecture Fully Linked.\n")

    def execute_comprehensive_diagnosis(self, patient_idx: int, image_path: str, mock_subject_id: str = "OAS2_0001"):
        """Executes full diagnostic pass across all 6 specialized entities."""
        print(f"--- Initiating Diagnostic Execution Loop: Patient Index {patient_idx} ---")
        
        # A. Gather Clinical Tabular Profiling
        patient_row = self.patient_df.iloc[patient_idx]
        mmse = float(patient_row['MMSE'])
        age = float(patient_row['Age'])
        
        # B. Gather Longitudinal Kinetics
        long_metrics = self.temporal_agent.calculate_progression_trajectory(mock_subject_id)
        atrophy_vel = float(long_metrics.get('atrophy_velocity_pct', 0.0))
        
        # C. Gather Vision Analytics
        img = Image.open(image_path).convert('L')
        raw_transformed = self.image_transform(img)
        assert isinstance(raw_transformed, torch.Tensor)
        img_tensor = raw_transformed.unsqueeze(0).to(self.device)
        
        # Enable gradient tracking for Grad-CAM processing
        img_tensor.requires_grad = True
        
        vision_output = self.vision_agent(img_tensor)
        probabilities = torch.nn.functional.softmax(vision_output[0], dim=0)
        pred_idx = int(torch.argmax(probabilities).item())
        pred_class = self.class_names[pred_idx]
        confidence = float(probabilities[pred_idx].item() * 100)
        
        # D. Execute Explainable AI Analysis
        heatmap = self.explainer_agent.generate_heatmap(img_tensor, target_class=pred_idx)
        peak_structural_focus = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        # E. Enforce Compliance & Ethical Guardrails
        is_flagged, restriction_log = self.ethicist_agent.audit_diagnostic_proposal(
            predicted_class=pred_class,
            confidence=confidence,
            mmse_score=mmse,
            atrophy_velocity=atrophy_vel
        )
        
        # F. Trigger Semantic Literature Search
        query = f"Clinical metrics for an MMSE of {mmse:.1f} and localized brain tissue changes."
        rag_output = self.rag_agent.query(query, top_k=1)[0][0]
        
        # G. Build Consolidated Diagnostic Output
        print("\n=======================================================")
        print("         SWARM CONSOLIDATED CLINICAL DIAGNOSIS         ")
        print("=======================================================")
        print(f"Patient Tracking ID :  {patient_row['Subject_ID']} (Longitudinal Profile: {mock_subject_id})")
        print(f"Chronological Age   :  {age:.1f} years")
        print(f"Cognitive Metrics   :  MMSE {mmse:.1f}/30.0")
        print("-------------------------------------------------------")
        print("LONGITUDINAL KINETICS (Agent 5):")
        print(f" > Monitored Timeline: {long_metrics.get('years_monitored', 'N/A')} years across {long_metrics.get('visits_tracked', 'N/A')} clinical visits.")
        print(f" > Atrophy Velocity  : {atrophy_vel:.3f}% whole brain structural loss / year.")
        print(f" > Progression Trend : {long_metrics.get('clinical_trend', 'N/A')}")
        print("-------------------------------------------------------")
        print("EXPLAINABLE RADIOMICS (Agents 1 & 4):")
        print(f" > Initial Prediction: {pred_class} (Model Confidence: {confidence:.2f}%)")
        print(f" > Grad-CAM Heatmap  : Localized peak feature density mapped to pixel tensor indices {peak_structural_focus}.")
        print("-------------------------------------------------------")
        print("COMPLIANCE & RISK MANAGEMENT REPORT (Agent 6):")
        if is_flagged:
            print(f" [CRITICAL ALARM] DIAGNOSIS OVERRIDDEN BY ETHICIST AGENT")
            print(f" > Action Taken      : Halted deployment. Routed case file to Human Neurologist.")
            print(f" > Infraction Reason : {restriction_log}")
        else:
            print(f" [STATUS: VERIFIED] {restriction_log}")
            print(f" > Authorized Diagnostic Classification: {pred_class}")
        print("-------------------------------------------------------")
        print("SUPPORTING MEDICAL LITERATURE (Agent 3):")
        print(f" > {rag_output}")
        print("=======================================================\n")

if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    cmo = AdvancedChiefMedicalOfficer(ROOT)
    
    # Locate a real dataset slice for the tracking validation test
    test_folder = os.path.join(ROOT, "data", "oasis_raw", "Non Demented")
    try:
        sample_img = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.jpeg', '.png'))][0]
        full_img_path = os.path.join(test_folder, sample_img)
        
        cmo.execute_comprehensive_diagnosis(patient_idx=0, image_path=full_img_path)
    except IndexError:
        print(f"[!] Target validation scan missing from directory path: {test_folder}")
         