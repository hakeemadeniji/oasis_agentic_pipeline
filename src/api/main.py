"""
FastAPI REST API for OASIS Agentic Pipeline
Production-ready API endpoints for multi-agent Alzheimer's diagnosis system.
"""

import os
import sys
import io
import base64
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch
import numpy as np
from PIL import Image
import uvicorn

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(src_dir)

from orchestrator.chief_medical_officer import AdvancedChiefMedicalOfficer
from api.heatmap import render_gradcam

# Initialize FastAPI app
app = FastAPI(
    title="OASIS Agentic Pipeline API",
    description="Multi-Agent AI System for Alzheimer's Disease Diagnosis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global CMO instance
ROOT_DIR = os.path.abspath(os.path.join(current_dir, '..', '..'))
cmo = None


# Pydantic models for request/response
class PatientData(BaseModel):
    """Patient clinical data"""
    patient_id: str = Field(..., description="Unique patient identifier")
    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    mmse: float = Field(..., ge=0, le=30, description="MMSE cognitive score")
    gender: Optional[str] = Field(None, description="Patient gender (M/F)")
    education: Optional[int] = Field(None, description="Years of education")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "OAS2_0001",
                "age": 75.5,
                "mmse": 24.0,
                "gender": "F",
                "education": 12
            }
        }


class DiagnosisRequest(BaseModel):
    """Complete diagnosis request"""
    patient_data: PatientData
    image_base64: Optional[str] = Field(None, description="Base64 encoded MRI image")
    longitudinal_id: Optional[str] = Field(None, description="Longitudinal tracking ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_data": {
                    "patient_id": "OAS2_0001",
                    "age": 75.5,
                    "mmse": 24.0
                },
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "longitudinal_id": "OAS2_0001"
            }
        }


class DiagnosisResponse(BaseModel):
    """Diagnosis result"""
    patient_id: str
    timestamp: str
    vision_prediction: Dict[str, Any]
    biomarker_analysis: Dict[str, Any]
    temporal_analysis: Dict[str, Any]
    rag_context: List[str]
    explainability: Dict[str, Any]
    ethics_audit: Dict[str, Any]
    regional_volumetry: Dict[str, Any] = {}
    atn_profile: Dict[str, Any] = {}
    clinical_narrative: str = ""
    reasoning_tier: str = ""
    final_diagnosis: str
    confidence: float
    approved: bool

    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "OAS2_0001",
                "timestamp": "2026-06-10T16:20:00Z",
                "vision_prediction": {
                    "class": "Very Mild Dementia",
                    "confidence": 87.5,
                    "probabilities": [0.05, 0.875, 0.05, 0.025]
                },
                "biomarker_analysis": {
                    "age_risk": "elevated",
                    "mmse_category": "mild_impairment"
                },
                "temporal_analysis": {
                    "trend": "Typical Age-Related Neuro-Degradation",
                    "atrophy_velocity": 0.45
                },
                "rag_context": ["Clinical guideline text..."],
                "explainability": {
                    "heatmap_available": True,
                    "peak_activation": [112, 98]
                },
                "ethics_audit": {
                    "approved": True,
                    "message": "All safety checks passed"
                },
                "final_diagnosis": "Very Mild Dementia",
                "confidence": 87.5,
                "approved": True
            }
        }


class BatchDiagnosisRequest(BaseModel):
    """Batch diagnosis request"""
    requests: List[DiagnosisRequest] = Field(..., max_length=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "requests": [
                    {
                        "patient_data": {
                            "patient_id": "OAS2_0001",
                            "age": 75.5,
                            "mmse": 24.0
                        }
                    }
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    device: str
    agents_loaded: bool
    timestamp: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize CMO on startup"""
    global cmo
    print("Initializing Chief Medical Officer...")
    cmo = AdvancedChiefMedicalOfficer(workspace_root=ROOT_DIR)
    print(f"✓ CMO initialized on device: {cmo.device}")


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health status"""
    return HealthResponse(
        status="healthy" if cmo is not None else "initializing",
        version="1.0.0",
        device=str(cmo.device) if cmo else "unknown",
        agents_loaded=cmo is not None,
        timestamp=datetime.utcnow().isoformat()
    )


# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """API root endpoint"""
    return {
        "message": "OASIS Agentic Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Single diagnosis endpoint
@app.post("/diagnose", response_model=DiagnosisResponse, tags=["Diagnosis"])
async def diagnose_patient(request: DiagnosisRequest):
    """
    Perform comprehensive multi-agent diagnosis for a single patient.
    
    - **patient_data**: Clinical patient information
    - **image_base64**: Optional base64 encoded MRI image
    - **longitudinal_id**: Optional ID for temporal analysis
    """
    if cmo is None:
        raise HTTPException(status_code=503, detail="CMO not initialized")
    
    try:
        # Process image if provided
        vision_result = None
        explainability_result = None
        
        if request.image_base64:
            # Decode base64 image
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('L')
            
            # Transform and predict
            img_tensor = cmo.image_transform(image).unsqueeze(0).to(cmo.device)
            
            with torch.no_grad():
                vision_output = cmo.vision_agent(img_tensor)
                probabilities = torch.nn.functional.softmax(vision_output[0], dim=0)
            
            pred_idx = int(torch.argmax(probabilities).item())
            pred_class = cmo.class_names[pred_idx]
            confidence = float(probabilities[pred_idx].item() * 100)
            
            vision_result = {
                "class": pred_class,
                "confidence": confidence,
                "probabilities": probabilities.cpu().numpy().tolist()
            }
            
            # Generate explainability heatmap (Grad-CAM) + colorized overlay layers.
            heatmap = cmo.explainer_agent.generate_heatmap(img_tensor, target_class=pred_idx)
            peak_focus = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            layers = render_gradcam(image, heatmap)

            explainability_result = {
                "heatmap_available": True,
                "peak_activation": [int(v) for v in peak_focus],
                "heatmap_shape": list(heatmap.shape),
                "base_png": layers["base"],
                "heatmap_png": layers["heatmap"],
                "overlay_png": layers["overlay"],
            }
        else:
            vision_result = {
                "class": "Not provided",
                "confidence": 0.0,
                "probabilities": []
            }
            explainability_result = {
                "heatmap_available": False
            }
        
        # Biomarker analysis
        biomarker_result = {
            "age_risk": "elevated" if request.patient_data.age > 70 else "normal",
            "mmse_category": "severe_impairment" if request.patient_data.mmse < 10 
                           else "moderate_impairment" if request.patient_data.mmse < 20
                           else "mild_impairment" if request.patient_data.mmse < 24
                           else "normal"
        }
        
        # Temporal analysis
        temporal_result = {"trend": "N/A", "atrophy_velocity": 0.0}
        if request.longitudinal_id:
            long_metrics = cmo.temporal_agent.calculate_progression_trajectory(request.longitudinal_id)
            temporal_result = {
                "trend": long_metrics.get('clinical_trend', 'N/A'),
                "atrophy_velocity": float(long_metrics.get('atrophy_velocity_pct', 0.0)),
                "visits_tracked": long_metrics.get('visits_tracked', 0),
                "years_monitored": long_metrics.get('years_monitored', 0)
            }
        
        # RAG query
        query = f"Clinical guidelines for MMSE {request.patient_data.mmse:.1f} and age {request.patient_data.age:.1f}"
        rag_results = cmo.rag_agent.query(query, top_k=2)
        rag_context = [result[0] for result in rag_results]
        
        # Ethics audit
        is_flagged, restriction_log = cmo.ethicist_agent.audit_diagnostic_proposal(
            predicted_class=vision_result["class"],
            confidence=vision_result["confidence"],
            mmse_score=request.patient_data.mmse,
            atrophy_velocity=temporal_result["atrophy_velocity"]
        )
        
        ethics_result = {
            "approved": not is_flagged,
            "message": restriction_log
        }

        # Regional volumetry (Agent 9): use FreeSurfer aseg when a subject id is given.
        volumetry_dict: Dict[str, Any] = {"source": "unavailable", "summary": "No FreeSurfer subject id provided."}
        volumetry_summary = volumetry_dict["summary"]
        hippo_zs: List[float] = []
        mta_risk = 0.0
        if request.longitudinal_id:
            vol = cmo.volumetry_agent.analyze_subject(request.longitudinal_id)
            volumetry_dict = vol.to_dict()
            volumetry_summary = vol.summary
            hippo_zs = [r.z_score for r in vol.regions if "Hippocampus" in r.structure]
            mta_risk = vol.mta_risk_score

        # ATN biomarker profile (Agent 10): A/T from OASIS-3 PET (PUP) SUVR when
        # available for the subject; N from regional volumetry.
        pet = cmo.pet_pup.analyze_subject(request.longitudinal_id) if request.longitudinal_id else None
        atn = cmo.atn_profiler.classify(
            amyloid_suvr=pet.amyloid_suvr if pet else None,
            amyloid_centiloid=pet.amyloid_centiloid if pet else None,
            amyloid_tracer=(pet.amyloid_tracer or "PIB") if pet else "PIB",
            tau_suvr=pet.tau_suvr if pet else None,
            hippocampus_z=(sum(hippo_zs) / len(hippo_zs)) if hippo_zs else None,
            mta_risk=mta_risk if mta_risk else None,
        )
        atn_dict = atn.to_dict()
        if pet:
            atn_dict["pet"] = pet.to_dict()

        # Final diagnosis
        final_diagnosis = vision_result["class"] if not is_flagged else "DIAGNOSIS_BLOCKED"

        # Hybrid edge-cloud clinical reasoner (Agent 8): grounded narrative via local Ollama.
        reasoning = cmo.reasoner_agent.synthesize({
            "prediction": vision_result["class"],
            "authorized_class": final_diagnosis,
            "confidence": vision_result["confidence"],
            "age": request.patient_data.age,
            "mmse": request.patient_data.mmse,
            "clinical_trend": temporal_result.get("trend", "N/A"),
            "atrophy_velocity": temporal_result.get("atrophy_velocity", 0.0),
            "volumetry_summary": volumetry_summary,
            "ethics_flagged": is_flagged,
            "ethics_message": restriction_log,
            "rag_context": rag_context,
        })

        return DiagnosisResponse(
            patient_id=request.patient_data.patient_id,
            timestamp=datetime.utcnow().isoformat(),
            vision_prediction=vision_result,
            biomarker_analysis=biomarker_result,
            temporal_analysis=temporal_result,
            rag_context=rag_context,
            explainability=explainability_result,
            ethics_audit=ethics_result,
            regional_volumetry=volumetry_dict,
            atn_profile=atn_dict,
            clinical_narrative=reasoning.narrative,
            reasoning_tier=f"{reasoning.tier}:{reasoning.model}",
            final_diagnosis=final_diagnosis,
            confidence=vision_result["confidence"],
            approved=not is_flagged
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnosis failed: {str(e)}")


# Batch diagnosis endpoint
@app.post("/diagnose/batch", tags=["Diagnosis"])
async def diagnose_batch(
    request: BatchDiagnosisRequest,
    background_tasks: BackgroundTasks
):
    """
    Perform batch diagnosis for multiple patients.
    
    - **requests**: List of diagnosis requests (max 100)
    
    Returns a job ID for tracking batch processing status.
    """
    if cmo is None:
        raise HTTPException(status_code=503, detail="CMO not initialized")
    
    if len(request.requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 requests per batch")
    
    # Generate job ID
    job_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    # Process batch in background
    async def process_batch():
        results = []
        for req in request.requests:
            try:
                result = await diagnose_patient(req)
                results.append(result.dict())
            except Exception as e:
                results.append({
                    "patient_id": req.patient_data.patient_id,
                    "error": str(e)
                })
        
        # Save results
        output_dir = Path(ROOT_DIR) / "data" / "batch_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_dir / f"{job_id}.json", 'w') as f:
            json.dump(results, f, indent=4)
    
    background_tasks.add_task(process_batch)
    
    return {
        "job_id": job_id,
        "status": "processing",
        "total_requests": len(request.requests),
        "message": "Batch processing started"
    }


# Image upload endpoint
@app.post("/upload/image", tags=["Upload"])
async def upload_image(file: UploadFile = File(...)):
    """
    Upload MRI image for diagnosis.
    
    Returns base64 encoded image for use in diagnosis request.
    """
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Encode to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "filename": file.filename,
            "size": len(contents),
            "format": image.format,
            "dimensions": image.size,
            "base64": img_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")


# Get patient data endpoint
@app.get("/patients/{patient_id}", tags=["Patients"])
async def get_patient(patient_id: str):
    """Get patient data by ID"""
    if cmo is None:
        raise HTTPException(status_code=503, detail="CMO not initialized")
    
    # Search in patient dataframe
    patient_row = cmo.patient_df[cmo.patient_df['Subject_ID'] == patient_id]
    
    if patient_row.empty:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    patient_data = patient_row.iloc[0].to_dict()
    return {
        "patient_id": patient_id,
        "data": patient_data
    }


# List patients endpoint
@app.get("/patients", tags=["Patients"])
async def list_patients(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List all patients with pagination"""
    if cmo is None:
        raise HTTPException(status_code=503, detail="CMO not initialized")
    
    total = len(cmo.patient_df)
    patients = cmo.patient_df.iloc[offset:offset+limit]
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "patients": patients.to_dict('records')
    }


# Model info endpoint
@app.get("/models/info", tags=["Models"])
async def model_info():
    """Get information about loaded models"""
    if cmo is None:
        raise HTTPException(status_code=503, detail="CMO not initialized")
    
    return {
        "vision_agent": {
            "architecture": "ResNet18",
            "input_shape": [1, 1, 224, 224],
            "num_classes": 4,
            "classes": cmo.class_names
        },
        "device": str(cmo.device),
        "agents": {
            "vision": "AlzheimerVisionAgent",
            "biomarker": "BiomarkerAgent",
            "rag": "RAGAgent",
            "explainer": "ExplainerAgent",
            "temporal": "TemporalAnalyst",
            "ethicist": "EthicistAgent",
            "reasoner": "ClinicalReasonerAgent (Ollama, hybrid edge-cloud)",
            "volumetry": "RegionalVolumetryAgent (FreeSurfer aseg)"
        },
        "acceleration": cmo.settings.onnx_providers,
        "llm": cmo.settings.summary()
    }


# Export results endpoint
@app.get("/export/{job_id}", tags=["Export"])
async def export_results(
    job_id: str,
    format: str = Query("json", regex="^(json|csv)$")
):
    """
    Export diagnosis results in specified format.
    
    - **job_id**: Batch job ID or patient ID
    - **format**: Export format (json or csv)
    """
    results_dir = Path(ROOT_DIR) / "data" / "batch_results"
    result_file = results_dir / f"{job_id}.json"
    
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    
    if format == "json":
        return FileResponse(
            result_file,
            media_type="application/json",
            filename=f"{job_id}.json"
        )
    elif format == "csv":
        # Convert JSON to CSV
        import json
        import csv
        
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Create CSV in memory
        output = io.StringIO()
        if isinstance(data, list) and len(data) > 0:
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={job_id}.csv"}
        )


# Statistics endpoint
@app.get("/stats", tags=["System"])
async def get_statistics():
    """Get system statistics"""
    if cmo is None:
        raise HTTPException(status_code=503, detail="CMO not initialized")
    
    return {
        "total_patients": len(cmo.patient_df),
        "device": str(cmo.device),
        "model_loaded": True,
        "agents_active": 8,
        "uptime": "N/A"  # Would need to track startup time
    }


# Sample MRI endpoint (one-click demo for the clinical console)
@app.get("/api/sample", tags=["Demo"])
async def sample_image(label: Optional[str] = Query(None, description="Class folder to sample from")):
    """
    Return a random OASIS MRI slice (base64 PNG) and its ground-truth class.

    Powers the 'Load sample scan' button in the clinical console so the UI is
    demonstrable on a hospital display without a local file to hand.
    """
    data_root = Path(ROOT_DIR) / "data" / "oasis_raw"
    if not data_root.exists():
        raise HTTPException(status_code=404, detail="OASIS dataset not found")

    class_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    if label:
        class_dirs = [d for d in class_dirs if d.name.lower() == label.lower()] or class_dirs
    if not class_dirs:
        raise HTTPException(status_code=404, detail="No class folders found")

    import random
    chosen_dir = random.choice(class_dirs)
    images = [p for p in chosen_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not images:
        raise HTTPException(status_code=404, detail="No images found")
    img_path = random.choice(images)

    image = Image.open(img_path).convert("L")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "true_label": chosen_dir.name,
        "filename": img_path.name,
        "image_base64": b64,
    }


# ---------------------------------------------------------------------------
# Production web console (static SPA). Mounted last so API routes take priority.
# Served at /app ; visiting / redirects there. Works on any browser/large display.
# ---------------------------------------------------------------------------
WEB_DIR = os.path.abspath(os.path.join(current_dir, "..", "..", "web"))
if os.path.isdir(WEB_DIR):
    app.mount("/app", StaticFiles(directory=WEB_DIR, html=True), name="console")

    @app.get("/console", include_in_schema=False)
    async def console_redirect():
        return RedirectResponse(url="/app/")


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
