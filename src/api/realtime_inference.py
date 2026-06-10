"""
Real-Time Inference Pipeline for OASIS Agentic Pipeline
WebSocket-based streaming inference with live updates.
"""

import os
import sys
import asyncio
import json
from typing import Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import base64
from io import BytesIO

import torch
import numpy as np
from PIL import Image
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import uvicorn

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(src_dir)

from orchestrator.chief_medical_officer import AdvancedChiefMedicalOfficer


class RealtimeInferenceEngine:
    """Real-time inference engine with streaming capabilities"""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        self.cmo = AdvancedChiefMedicalOfficer(workspace_root=workspace_root)
        self.active_connections: Dict[str, WebSocket] = {}
        
        print(f"✓ Real-time Inference Engine initialized")
        print(f"  Device: {self.cmo.device}")
    
    async def connect_websocket(self, websocket: WebSocket, client_id: str):
        """Connect a new WebSocket client"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"✓ Client {client_id} connected")
    
    def disconnect_websocket(self, client_id: str):
        """Disconnect a WebSocket client"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"✓ Client {client_id} disconnected")
    
    async def send_progress_update(
        self,
        websocket: WebSocket,
        stage: str,
        progress: float,
        message: str
    ):
        """Send progress update to client"""
        update = {
            "type": "progress",
            "stage": stage,
            "progress": progress,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        await websocket.send_json(update)
    
    async def stream_diagnosis(
        self,
        websocket: WebSocket,
        patient_data: Dict[str, Any],
        image_base64: Optional[str] = None
    ):
        """Stream diagnosis results in real-time"""
        try:
            patient_id = patient_data.get('patient_id', 'UNKNOWN')
            age = float(patient_data.get('age', 0))
            mmse = float(patient_data.get('mmse', 0))
            
            # Stage 1: Image Processing
            await self.send_progress_update(
                websocket,
                "image_processing",
                0.1,
                "Processing MRI image..."
            )
            
            vision_result = None
            explainability_result = None
            
            if image_base64:
                # Decode and process image
                image_data = base64.b64decode(image_base64)
                img = Image.open(BytesIO(image_data)).convert('L')
                img_tensor = self.cmo.image_transform(img).unsqueeze(0).to(self.cmo.device)
                
                await self.send_progress_update(
                    websocket,
                    "vision_analysis",
                    0.3,
                    "Running Vision Agent analysis..."
                )
                
                # Vision Agent prediction
                with torch.no_grad():
                    vision_output = self.cmo.vision_agent(img_tensor)
                    probabilities = torch.nn.functional.softmax(vision_output[0], dim=0)
                
                pred_idx = int(torch.argmax(probabilities).item())
                pred_class = self.cmo.class_names[pred_idx]
                confidence = float(probabilities[pred_idx].item() * 100)
                
                vision_result = {
                    "class": pred_class,
                    "confidence": confidence,
                    "probabilities": probabilities.cpu().numpy().tolist()
                }
                
                # Send vision results
                await websocket.send_json({
                    "type": "result",
                    "agent": "vision",
                    "data": vision_result
                })
                
                await self.send_progress_update(
                    websocket,
                    "explainability",
                    0.5,
                    "Generating Grad-CAM heatmap..."
                )
                
                # Generate explainability
                img_tensor.requires_grad = True
                heatmap = self.cmo.explainer_agent.generate_heatmap(img_tensor, target_class=pred_idx)
                peak_focus = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                
                explainability_result = {
                    "heatmap_available": True,
                    "peak_activation": list(peak_focus),
                    "heatmap_shape": list(heatmap.shape)
                }
                
                # Send explainability results
                await websocket.send_json({
                    "type": "result",
                    "agent": "explainer",
                    "data": explainability_result
                })
            else:
                vision_result = {"class": "Not provided", "confidence": 0.0}
                explainability_result = {"heatmap_available": False}
            
            # Stage 2: Temporal Analysis
            await self.send_progress_update(
                websocket,
                "temporal_analysis",
                0.6,
                "Analyzing longitudinal progression..."
            )
            
            longitudinal_id = patient_data.get('longitudinal_id')
            if longitudinal_id:
                long_metrics = self.cmo.temporal_agent.calculate_progression_trajectory(longitudinal_id)
                temporal_result = {
                    "trend": long_metrics.get('clinical_trend', 'N/A'),
                    "atrophy_velocity": float(long_metrics.get('atrophy_velocity_pct', 0.0)),
                    "visits_tracked": long_metrics.get('visits_tracked', 0),
                    "years_monitored": long_metrics.get('years_monitored', 0)
                }
            else:
                temporal_result = {"trend": "N/A", "atrophy_velocity": 0.0}
            
            # Send temporal results
            await websocket.send_json({
                "type": "result",
                "agent": "temporal",
                "data": temporal_result
            })
            
            # Stage 3: RAG Query
            await self.send_progress_update(
                websocket,
                "rag_query",
                0.75,
                "Retrieving clinical guidelines..."
            )
            
            query = f"Clinical guidelines for MMSE {mmse:.1f} and age {age:.1f}"
            rag_results = self.cmo.rag_agent.query(query, top_k=2)
            rag_context = [result[0] for result in rag_results]
            
            # Send RAG results
            await websocket.send_json({
                "type": "result",
                "agent": "rag",
                "data": {"context": rag_context}
            })
            
            # Stage 4: Ethics Audit
            await self.send_progress_update(
                websocket,
                "ethics_audit",
                0.9,
                "Performing ethics and safety audit..."
            )
            
            is_flagged, restriction_log = self.cmo.ethicist_agent.audit_diagnostic_proposal(
                predicted_class=vision_result["class"],
                confidence=vision_result["confidence"],
                mmse_score=mmse,
                atrophy_velocity=temporal_result["atrophy_velocity"]
            )
            
            ethics_result = {
                "approved": not is_flagged,
                "message": restriction_log
            }
            
            # Send ethics results
            await websocket.send_json({
                "type": "result",
                "agent": "ethics",
                "data": ethics_result
            })
            
            # Stage 5: Final Diagnosis
            await self.send_progress_update(
                websocket,
                "finalization",
                1.0,
                "Finalizing diagnosis..."
            )
            
            final_diagnosis = {
                "patient_id": patient_id,
                "timestamp": datetime.utcnow().isoformat(),
                "diagnosis": vision_result["class"] if not is_flagged else "BLOCKED",
                "confidence": vision_result["confidence"],
                "approved": not is_flagged,
                "vision_prediction": vision_result,
                "temporal_analysis": temporal_result,
                "rag_context": rag_context,
                "explainability": explainability_result,
                "ethics_audit": ethics_result
            }
            
            # Send final diagnosis
            await websocket.send_json({
                "type": "final",
                "data": final_diagnosis
            })
            
        except Exception as e:
            # Send error
            await websocket.send_json({
                "type": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def stream_batch_diagnosis(
        self,
        websocket: WebSocket,
        patients: list
    ):
        """Stream batch diagnosis results"""
        total = len(patients)
        
        for idx, patient_data in enumerate(patients):
            try:
                # Send batch progress
                await websocket.send_json({
                    "type": "batch_progress",
                    "current": idx + 1,
                    "total": total,
                    "progress": (idx + 1) / total,
                    "patient_id": patient_data.get('patient_id', f'P{idx}')
                })
                
                # Process patient
                await self.stream_diagnosis(
                    websocket,
                    patient_data,
                    patient_data.get('image_base64')
                )
                
                # Small delay between patients
                await asyncio.sleep(0.1)
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "patient_id": patient_data.get('patient_id', f'P{idx}'),
                    "message": str(e)
                })
        
        # Send batch complete
        await websocket.send_json({
            "type": "batch_complete",
            "total_processed": total,
            "timestamp": datetime.utcnow().isoformat()
        })


class StreamingInferenceAPI:
    """Streaming API for Server-Sent Events (SSE)"""
    
    def __init__(self, workspace_root: str):
        self.engine = RealtimeInferenceEngine(workspace_root)
    
    async def stream_diagnosis_sse(
        self,
        patient_data: Dict[str, Any],
        image_base64: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream diagnosis results as Server-Sent Events"""
        
        # Helper to format SSE message
        def format_sse(data: Dict[str, Any]) -> str:
            return f"data: {json.dumps(data)}\n\n"
        
        try:
            patient_id = patient_data.get('patient_id', 'UNKNOWN')
            age = float(patient_data.get('age', 0))
            mmse = float(patient_data.get('mmse', 0))
            
            # Stage 1: Image Processing
            yield format_sse({
                "type": "progress",
                "stage": "image_processing",
                "progress": 0.1,
                "message": "Processing MRI image..."
            })
            
            vision_result = None
            
            if image_base64:
                image_data = base64.b64decode(image_base64)
                img = Image.open(BytesIO(image_data)).convert('L')
                img_tensor = self.engine.cmo.image_transform(img).unsqueeze(0).to(self.engine.cmo.device)
                
                yield format_sse({
                    "type": "progress",
                    "stage": "vision_analysis",
                    "progress": 0.3,
                    "message": "Running Vision Agent..."
                })
                
                with torch.no_grad():
                    vision_output = self.engine.cmo.vision_agent(img_tensor)
                    probabilities = torch.nn.functional.softmax(vision_output[0], dim=0)
                
                pred_idx = int(torch.argmax(probabilities).item())
                pred_class = self.engine.cmo.class_names[pred_idx]
                confidence = float(probabilities[pred_idx].item() * 100)
                
                vision_result = {
                    "class": pred_class,
                    "confidence": confidence,
                    "probabilities": probabilities.cpu().numpy().tolist()
                }
                
                yield format_sse({
                    "type": "result",
                    "agent": "vision",
                    "data": vision_result
                })
            else:
                vision_result = {"class": "Not provided", "confidence": 0.0}
            
            # Stage 2: Temporal Analysis
            yield format_sse({
                "type": "progress",
                "stage": "temporal_analysis",
                "progress": 0.6,
                "message": "Analyzing temporal progression..."
            })
            
            longitudinal_id = patient_data.get('longitudinal_id')
            if longitudinal_id:
                long_metrics = self.engine.cmo.temporal_agent.calculate_progression_trajectory(longitudinal_id)
                temporal_result = {
                    "trend": long_metrics.get('clinical_trend', 'N/A'),
                    "atrophy_velocity": float(long_metrics.get('atrophy_velocity_pct', 0.0))
                }
            else:
                temporal_result = {"trend": "N/A", "atrophy_velocity": 0.0}
            
            yield format_sse({
                "type": "result",
                "agent": "temporal",
                "data": temporal_result
            })
            
            # Stage 3: Ethics Audit
            yield format_sse({
                "type": "progress",
                "stage": "ethics_audit",
                "progress": 0.9,
                "message": "Performing safety audit..."
            })
            
            is_flagged, restriction_log = self.engine.cmo.ethicist_agent.audit_diagnostic_proposal(
                predicted_class=vision_result["class"],
                confidence=vision_result["confidence"],
                mmse_score=mmse,
                atrophy_velocity=temporal_result["atrophy_velocity"]
            )
            
            # Final result
            yield format_sse({
                "type": "final",
                "data": {
                    "patient_id": patient_id,
                    "diagnosis": vision_result["class"] if not is_flagged else "BLOCKED",
                    "confidence": vision_result["confidence"],
                    "approved": not is_flagged,
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
            
        except Exception as e:
            yield format_sse({
                "type": "error",
                "message": str(e)
            })


# WebSocket endpoint handler
async def websocket_diagnosis_handler(
    websocket: WebSocket,
    engine: RealtimeInferenceEngine,
    client_id: str
):
    """Handle WebSocket diagnosis requests"""
    await engine.connect_websocket(websocket, client_id)
    
    try:
        while True:
            # Receive request
            data = await websocket.receive_json()
            
            request_type = data.get('type')
            
            if request_type == 'diagnose':
                # Single patient diagnosis
                patient_data = data.get('patient_data', {})
                image_base64 = data.get('image_base64')
                
                await engine.stream_diagnosis(websocket, patient_data, image_base64)
                
            elif request_type == 'batch':
                # Batch diagnosis
                patients = data.get('patients', [])
                await engine.stream_batch_diagnosis(websocket, patients)
                
            elif request_type == 'ping':
                # Heartbeat
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        engine.disconnect_websocket(client_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        engine.disconnect_websocket(client_id)


def main():
    """Main entry point for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Inference Engine')
    parser.add_argument('--workspace', default='.', help='Workspace root directory')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = RealtimeInferenceEngine(workspace_root=args.workspace)
    
    print("\n✓ Real-time Inference Engine ready")
    print("  Use with FastAPI WebSocket endpoints")


if __name__ == '__main__':
    main()
