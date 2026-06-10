import os
import sys
import torch
import numpy as np
import streamlit as st
from PIL import Image

# Synchronize directory structure across application layers
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(src_dir)

from orchestrator.chief_medical_officer import AdvancedChiefMedicalOfficer

# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="OASIS Cognitive Multi-Agent Swarm",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHE THE AGENT SWARM INITIALIZATION ---
@st.cache_resource
def bootstrap_cmo_swarm(root_dir: str):
    return AdvancedChiefMedicalOfficer(workspace_root=root_dir)

# Resolve project absolute path
ROOT_DIR = os.path.abspath(os.path.join(current_dir, '..', '..'))
cmo = bootstrap_cmo_swarm(ROOT_DIR)

st.title("🧠 OASIS Multimodal Cognitive Swarm Dashboard")
st.caption("Production-grade Clinical Multi-Agent Framework optimized for ARM64 Architecture.")
st.markdown("---")

# --- SIDEBAR: PATIENT SELECTION & OVERRIDE SANDBOX ---
st.sidebar.header("🔬 Patient Data Controller")

# Patient Cohort Selection (0-99 from the parsed tabular CSV)
patient_idx = st.sidebar.selectbox(
    "Select Cohort Patient ID",
    options=list(range(len(cmo.patient_df))),
    format_func=lambda x: f"{cmo.patient_df.iloc[x]['Subject_ID']} (Index {x})"
)

# Extract real data point values
real_row = cmo.patient_df.iloc[patient_idx]
real_age = float(real_row['Age'])
real_mmse = float(real_row['MMSE'])

st.sidebar.markdown("### 🎛️ Sandbox Override Controls")
st.sidebar.info("Modify these values to stress-test Agent 6's Ethical Guardrail Audit mechanisms in real-time.")

# Real-time data override sliders
override_active = st.sidebar.checkbox("Activate Sandbox Parameter Override", value=False)
if override_active:
    target_age = st.sidebar.slider("Override Chronological Age", 60.0, 100.0, real_age, 0.1)
    target_mmse = st.sidebar.slider("Override Cognitive MMSE Score", 0.0, 30.0, real_mmse, 0.1)
else:
    target_age = real_age
    target_mmse = real_mmse

st.sidebar.markdown("---")
st.sidebar.markdown("**System Metrics:**")
# FIXED: Cast the torch.device object to a string before calling .upper()
st.sidebar.text(f"Compute Hardware: {str(cmo.device).upper()}")

# --- DATA FUSION & COMPUTATION PHASE ---
mock_long_id = "OAS2_0001" if patient_idx % 2 == 0 else "OAS2_0002"
long_metrics = cmo.temporal_agent.calculate_progression_trajectory(mock_long_id)
atrophy_vel = float(long_metrics.get('atrophy_velocity_pct', 0.0))

# Extract Raw Image Target
test_folder = os.path.join(ROOT_DIR, "data", "oasis_raw", "Non Demented")
try:
    sample_images = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    selected_image_file = sample_images[patient_idx % len(sample_images)]
    image_path = os.path.join(test_folder, selected_image_file)
except IndexError:
    image_path = None

# --- EXECUTE MULTIMODAL INFERENCE PASS ---
if image_path and os.path.exists(image_path):
    img = Image.open(image_path).convert('L')
    
    # FIXED: Explicitly assert type properties to unlock gradient tracking on the interface layer
    raw_transformed = cmo.image_transform(img)
    assert isinstance(raw_transformed, torch.Tensor)
    img_tensor = raw_transformed.unsqueeze(0).to(cmo.device)
    img_tensor.requires_grad = True
    
    # Vision Agent forward prediction pass
    with torch.enable_grad():
        vision_output = cmo.vision_agent(img_tensor)
        probabilities = torch.nn.functional.softmax(vision_output[0], dim=0)
        
    pred_idx = int(torch.argmax(probabilities).item())
    pred_class = cmo.class_names[pred_idx]
    confidence = float(probabilities[pred_idx].item() * 100)
    
    # Render XAI Heatmap mapping matrix
    heatmap = cmo.explainer_agent.generate_heatmap(img_tensor, target_class=pred_idx)
    peak_focus = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    # Force Ethicist Audit on active parameters
    is_flagged, restriction_log = cmo.ethicist_agent.audit_diagnostic_proposal(
        predicted_class=pred_class,
        confidence=confidence,
        mmse_score=target_mmse,
        atrophy_velocity=atrophy_vel
    )
    
    # Query RAG Engine
    query = f"Clinical guidelines for MMSE {target_mmse:.1f} and brain class {pred_class}."
    rag_context = cmo.rag_agent.query(query, top_k=1)[0][0]
    
    # --- UI INTERFACE RENDERING LAYOUT ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Patient Target Profile", value=real_row['Subject_ID'], delta=f"Index Map {patient_idx}")
    with col2:
        st.metric(label="Chronological Age Vector", value=f"{target_age:.1f} Yrs", delta="Ingested Tabular" if not override_active else "Sandbox Overridden", delta_color="inverse" if override_active else "normal")
    with col3:
        st.metric(label="Cognitive Screening Evaluation", value=f"{target_mmse:.1f} / 30.0", delta="Baseline MMSE" if not override_active else "Sandbox Overridden", delta_color="inverse" if override_active else "normal")
        
    st.markdown("### 🚨 AGENT 6: Compliance & Risk Management Audit")
    if is_flagged:
        st.error(f"**CRITICAL ALARM: DEPLOYMENT OVERRIDDEN**\n\n**Reason:** {restriction_log}\n\n*Action Taken: Automated security execution has blocked this diagnosis and routed the file to a senior clinical neurologist.*")
    else:
        st.success(f"**VERIFIED COMPLIANCE STATUS:** {restriction_log}\n\n*Authorized Diagnostic Vector:* **{pred_class}**")
        
    st.markdown("---")
    
    left_panel, right_panel = st.columns([1, 1])
    
    with left_panel:
        st.subheader("🖼️ Explainable Radiomics Engine (Agents 1 & 4)")
        st.image(image_path, caption=f"Active Raw Scan Target: {selected_image_file}", use_container_width=True)
        
        st.markdown(f"**Spatial Vision Assessment:** `{pred_class}`")
        st.progress(confidence / 100.0)
        st.caption(f"Raw Softmax Probability Confidence: {confidence:.2f}%")
        st.caption(f"**Grad-CAM Feature Activation Target Mapping:** High-density structural feature focus locked to matrix coordinates: {peak_focus}")
        
    with right_panel:
        st.subheader("📈 Longitudinal Analytics (Agent 5)")
        st.markdown(f"**Calculated Trend Profile:** `{long_metrics.get('clinical_trend', 'N/A')}`")
        
        st.info(f"""
        * **Historical Tracking Record:** {long_metrics.get('visits_tracked', 'N/A')} distinct tracking checkpoints mapped across {long_metrics.get('years_monitored', 'N/A')} validation years.
        * **Brain Mass Atrophy Velocity:** **{atrophy_vel:.3f}%** structural volume degradation tracking / year.
        * **Cognitive Drift Velocity:** {long_metrics.get('mmse_drift', 'N/A')} point movement on standard evaluation index.
        """)
        
        st.subheader("📚 Medical Librarian RAG System (Agent 3)")
        st.markdown("**Semantically Retrieved Clinical Context:**")
        st.write(f"> *{rag_context}*")
        st.caption(f"Model Source Backbone: sentence-transformers/all-MiniLM-L6-v2 (Cached in RAM)")

else:
    st.error(f"[!] System Error: Failed to cleanly route patient data directories at target location: `{test_folder}`")