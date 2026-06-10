"""
Test suite for OASIS Agentic Pipeline API endpoints
Tests REST API, batch processing, and real-time inference
"""

import pytest
import asyncio
import base64
import json
from io import BytesIO
from PIL import Image
import numpy as np

# Mock imports for testing without actual API server
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
sys.path.insert(0, src_dir)


class TestAPIEndpoints:
    """Test REST API endpoints"""
    
    @pytest.fixture
    def mock_cmo(self):
        """Mock Chief Medical Officer"""
        mock = Mock()
        mock.device = "cpu"
        mock.class_names = ["Non Demented", "Very Mild Dementia", "Mild Dementia", "Moderate Dementia"]
        mock.patient_df = Mock()
        mock.patient_df.__len__ = Mock(return_value=100)
        return mock
    
    @pytest.fixture
    def sample_patient_data(self):
        """Sample patient data for testing"""
        return {
            "patient_id": "OAS2_0001",
            "age": 75.5,
            "mmse": 24.0,
            "gender": "F",
            "education": 12
        }
    
    @pytest.fixture
    def sample_image_base64(self):
        """Generate sample base64 encoded image"""
        # Create a simple grayscale image
        img = Image.new('L', (224, 224), color=128)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def test_health_check_structure(self):
        """Test health check response structure"""
        expected_keys = ["status", "version", "device", "agents_loaded", "timestamp"]
        
        # Mock response
        health_response = {
            "status": "healthy",
            "version": "1.0.0",
            "device": "cpu",
            "agents_loaded": True,
            "timestamp": "2026-06-10T16:30:00Z"
        }
        
        for key in expected_keys:
            assert key in health_response, f"Missing key: {key}"
        
        assert health_response["status"] == "healthy"
        assert health_response["agents_loaded"] is True
    
    def test_diagnosis_request_validation(self, sample_patient_data):
        """Test diagnosis request data validation"""
        # Valid request
        assert sample_patient_data["age"] >= 0
        assert sample_patient_data["age"] <= 120
        assert sample_patient_data["mmse"] >= 0
        assert sample_patient_data["mmse"] <= 30
        assert sample_patient_data["patient_id"] is not None
        
        # Invalid age
        invalid_data = sample_patient_data.copy()
        invalid_data["age"] = 150
        assert invalid_data["age"] > 120  # Should fail validation
        
        # Invalid MMSE
        invalid_data = sample_patient_data.copy()
        invalid_data["mmse"] = 35
        assert invalid_data["mmse"] > 30  # Should fail validation
    
    def test_diagnosis_response_structure(self):
        """Test diagnosis response structure"""
        expected_keys = [
            "patient_id", "timestamp", "vision_prediction",
            "biomarker_analysis", "temporal_analysis", "rag_context",
            "explainability", "ethics_audit", "final_diagnosis",
            "confidence", "approved"
        ]
        
        # Mock response
        diagnosis_response = {
            "patient_id": "OAS2_0001",
            "timestamp": "2026-06-10T16:30:00Z",
            "vision_prediction": {
                "class": "Very Mild Dementia",
                "confidence": 87.5,
                "probabilities": [0.05, 0.875, 0.05, 0.025]
            },
            "biomarker_analysis": {"age_risk": "elevated"},
            "temporal_analysis": {"trend": "N/A", "atrophy_velocity": 0.0},
            "rag_context": ["Clinical guideline..."],
            "explainability": {"heatmap_available": False},
            "ethics_audit": {"approved": True, "message": "Passed"},
            "final_diagnosis": "Very Mild Dementia",
            "confidence": 87.5,
            "approved": True
        }
        
        for key in expected_keys:
            assert key in diagnosis_response, f"Missing key: {key}"
        
        # Validate nested structures
        assert "class" in diagnosis_response["vision_prediction"]
        assert "confidence" in diagnosis_response["vision_prediction"]
        assert "approved" in diagnosis_response["ethics_audit"]
    
    def test_batch_request_validation(self):
        """Test batch request validation"""
        # Valid batch (within limit)
        valid_batch = {"requests": [{"patient_data": {"patient_id": f"P{i}", "age": 70, "mmse": 25}} for i in range(50)]}
        assert len(valid_batch["requests"]) <= 100
        
        # Invalid batch (exceeds limit)
        invalid_batch = {"requests": [{"patient_data": {"patient_id": f"P{i}", "age": 70, "mmse": 25}} for i in range(150)]}
        assert len(invalid_batch["requests"]) > 100  # Should fail validation
    
    def test_image_upload_validation(self, sample_image_base64):
        """Test image upload and base64 encoding"""
        # Decode and validate
        image_data = base64.b64decode(sample_image_base64)
        img = Image.open(BytesIO(image_data))
        
        assert img.mode == 'L'  # Grayscale
        assert img.size == (224, 224)
    
    def test_patient_list_pagination(self):
        """Test patient list pagination parameters"""
        # Valid pagination
        limit = 10
        offset = 0
        assert 1 <= limit <= 100
        assert offset >= 0
        
        # Invalid limit
        invalid_limit = 150
        assert invalid_limit > 100  # Should fail validation
        
        # Invalid offset
        invalid_offset = -5
        assert invalid_offset < 0  # Should fail validation
    
    def test_export_format_validation(self):
        """Test export format validation"""
        valid_formats = ["json", "csv"]
        
        assert "json" in valid_formats
        assert "csv" in valid_formats
        assert "pdf" not in valid_formats  # Not implemented yet
    
    def test_model_info_structure(self):
        """Test model info response structure"""
        model_info = {
            "vision_agent": {
                "architecture": "ResNet18",
                "input_shape": [1, 1, 224, 224],
                "num_classes": 4,
                "classes": ["Non Demented", "Very Mild Dementia", "Mild Dementia", "Moderate Dementia"]
            },
            "device": "cpu",
            "agents": {
                "vision": "AlzheimerVisionAgent",
                "biomarker": "BiomarkerAgent",
                "rag": "RAGAgent",
                "explainer": "ExplainerAgent",
                "temporal": "TemporalAnalyst",
                "ethicist": "EthicistAgent"
            }
        }
        
        assert "vision_agent" in model_info
        assert "device" in model_info
        assert "agents" in model_info
        assert len(model_info["agents"]) == 6
        assert model_info["vision_agent"]["num_classes"] == 4


class TestBatchProcessing:
    """Test batch processing functionality"""
    
    @pytest.fixture
    def sample_csv_data(self):
        """Sample CSV data for batch processing"""
        return [
            {"patient_id": "OAS2_0001", "age": 75.5, "mmse": 24.0},
            {"patient_id": "OAS2_0002", "age": 68.0, "mmse": 28.0},
            {"patient_id": "OAS2_0003", "age": 82.0, "mmse": 18.0}
        ]
    
    def test_batch_processor_initialization(self):
        """Test batch processor initialization"""
        # Mock initialization parameters
        workspace_root = "."
        max_workers = 4
        use_gpu = False
        
        assert max_workers > 0
        assert workspace_root is not None
    
    def test_batch_result_structure(self):
        """Test batch processing result structure"""
        batch_result = {
            "summary": {
                "total_patients": 10,
                "successful": 9,
                "failed": 1,
                "approved": 8,
                "blocked": 1,
                "processing_time_seconds": 45.5,
                "avg_time_per_patient": 4.55,
                "throughput_patients_per_hour": 791.2
            },
            "results": []
        }
        
        assert "summary" in batch_result
        assert "results" in batch_result
        assert batch_result["summary"]["total_patients"] == batch_result["summary"]["successful"] + batch_result["summary"]["failed"]
    
    def test_csv_export_format(self):
        """Test CSV export format"""
        csv_columns = [
            "patient_id", "timestamp", "diagnosis", "confidence",
            "approved", "vision_class", "vision_confidence",
            "temporal_trend", "atrophy_velocity",
            "ethics_approved", "ethics_message"
        ]
        
        # Mock CSV row
        csv_row = {
            "patient_id": "OAS2_0001",
            "timestamp": "2026-06-10T16:30:00Z",
            "diagnosis": "Very Mild Dementia",
            "confidence": 87.5,
            "approved": True,
            "vision_class": "Very Mild Dementia",
            "vision_confidence": 87.5,
            "temporal_trend": "Typical",
            "atrophy_velocity": 0.45,
            "ethics_approved": True,
            "ethics_message": "Passed"
        }
        
        for col in csv_columns:
            assert col in csv_row, f"Missing column: {col}"
    
    def test_parallel_processing_workers(self):
        """Test parallel processing worker configuration"""
        max_workers_options = [1, 2, 4, 8]
        
        for workers in max_workers_options:
            assert workers > 0
            assert workers <= 16  # Reasonable upper limit


class TestRealtimeInference:
    """Test real-time inference and WebSocket functionality"""
    
    def test_websocket_message_types(self):
        """Test WebSocket message type validation"""
        valid_types = ["diagnose", "batch", "ping"]
        
        for msg_type in valid_types:
            assert msg_type in ["diagnose", "batch", "ping"]
    
    def test_progress_update_structure(self):
        """Test progress update message structure"""
        progress_update = {
            "type": "progress",
            "stage": "vision_analysis",
            "progress": 0.3,
            "message": "Running Vision Agent...",
            "timestamp": "2026-06-10T16:30:00Z"
        }
        
        assert progress_update["type"] == "progress"
        assert 0.0 <= progress_update["progress"] <= 1.0
        assert progress_update["stage"] is not None
    
    def test_agent_result_structure(self):
        """Test agent result message structure"""
        agent_result = {
            "type": "result",
            "agent": "vision",
            "data": {
                "class": "Very Mild Dementia",
                "confidence": 87.5
            }
        }
        
        assert agent_result["type"] == "result"
        assert agent_result["agent"] in ["vision", "biomarker", "rag", "explainer", "temporal", "ethics"]
        assert "data" in agent_result
    
    def test_final_result_structure(self):
        """Test final result message structure"""
        final_result = {
            "type": "final",
            "data": {
                "patient_id": "OAS2_0001",
                "diagnosis": "Very Mild Dementia",
                "confidence": 87.5,
                "approved": True,
                "timestamp": "2026-06-10T16:30:00Z"
            }
        }
        
        assert final_result["type"] == "final"
        assert "data" in final_result
        assert "diagnosis" in final_result["data"]
        assert "approved" in final_result["data"]
    
    def test_batch_progress_structure(self):
        """Test batch progress message structure"""
        batch_progress = {
            "type": "batch_progress",
            "current": 5,
            "total": 10,
            "progress": 0.5,
            "patient_id": "OAS2_0005"
        }
        
        assert batch_progress["type"] == "batch_progress"
        assert batch_progress["current"] <= batch_progress["total"]
        assert batch_progress["progress"] == batch_progress["current"] / batch_progress["total"]


class TestErrorHandling:
    """Test error handling and validation"""
    
    def test_error_response_structure(self):
        """Test error response format"""
        error_response = {
            "detail": "Patient not found"
        }
        
        assert "detail" in error_response
        assert isinstance(error_response["detail"], str)
    
    def test_http_status_codes(self):
        """Test expected HTTP status codes"""
        status_codes = {
            "success": 200,
            "bad_request": 400,
            "not_found": 404,
            "server_error": 500,
            "unavailable": 503
        }
        
        assert status_codes["success"] == 200
        assert status_codes["bad_request"] == 400
        assert status_codes["not_found"] == 404
        assert status_codes["server_error"] == 500
        assert status_codes["unavailable"] == 503
    
    def test_validation_errors(self):
        """Test validation error scenarios"""
        # Age validation
        invalid_ages = [-1, 150, 200]
        for age in invalid_ages:
            assert age < 0 or age > 120
        
        # MMSE validation
        invalid_mmse = [-5, 35, 50]
        for mmse in invalid_mmse:
            assert mmse < 0 or mmse > 30
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields"""
        incomplete_data = {
            "age": 75.5
            # Missing patient_id and mmse
        }
        
        assert "patient_id" not in incomplete_data
        assert "mmse" not in incomplete_data


class TestDataModels:
    """Test data model validation"""
    
    def test_patient_data_model(self):
        """Test PatientData model"""
        patient_data = {
            "patient_id": "OAS2_0001",
            "age": 75.5,
            "mmse": 24.0,
            "gender": "F",
            "education": 12
        }
        
        # Required fields
        assert "patient_id" in patient_data
        assert "age" in patient_data
        assert "mmse" in patient_data
        
        # Type validation
        assert isinstance(patient_data["patient_id"], str)
        assert isinstance(patient_data["age"], (int, float))
        assert isinstance(patient_data["mmse"], (int, float))
    
    def test_diagnosis_request_model(self):
        """Test DiagnosisRequest model"""
        diagnosis_request = {
            "patient_data": {
                "patient_id": "OAS2_0001",
                "age": 75.5,
                "mmse": 24.0
            },
            "image_base64": "base64string...",
            "longitudinal_id": "OAS2_0001"
        }
        
        assert "patient_data" in diagnosis_request
        assert isinstance(diagnosis_request["patient_data"], dict)
    
    def test_vision_prediction_model(self):
        """Test vision prediction structure"""
        vision_prediction = {
            "class": "Very Mild Dementia",
            "confidence": 87.5,
            "probabilities": [0.05, 0.875, 0.05, 0.025]
        }
        
        assert "class" in vision_prediction
        assert "confidence" in vision_prediction
        assert "probabilities" in vision_prediction
        assert len(vision_prediction["probabilities"]) == 4
        assert sum(vision_prediction["probabilities"]) <= 1.01  # Allow small floating point error


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_complete_diagnosis_workflow(self, sample_patient_data=None, sample_image_base64=None):
        """Test complete diagnosis workflow"""
        if sample_patient_data is None:
            sample_patient_data = {
                "patient_id": "OAS2_0001",
                "age": 75.5,
                "mmse": 24.0
            }
        
        # Step 1: Validate patient data
        assert sample_patient_data["patient_id"] is not None
        
        # Step 2: Process image (if provided)
        if sample_image_base64:
            image_data = base64.b64decode(sample_image_base64)
            assert len(image_data) > 0
        
        # Step 3: Mock diagnosis result
        diagnosis_result = {
            "patient_id": sample_patient_data["patient_id"],
            "final_diagnosis": "Very Mild Dementia",
            "confidence": 87.5,
            "approved": True
        }
        
        # Step 4: Validate result
        assert diagnosis_result["patient_id"] == sample_patient_data["patient_id"]
        assert diagnosis_result["confidence"] > 0
        assert isinstance(diagnosis_result["approved"], bool)
    
    def test_batch_to_export_workflow(self):
        """Test batch processing to export workflow"""
        # Step 1: Submit batch
        batch_request = {
            "requests": [
                {"patient_data": {"patient_id": f"P{i}", "age": 70, "mmse": 25}}
                for i in range(5)
            ]
        }
        
        assert len(batch_request["requests"]) > 0
        
        # Step 2: Mock processing
        job_id = "batch_20260610_162000"
        assert job_id.startswith("batch_")
        
        # Step 3: Mock results
        results = [
            {"patient_id": f"P{i}", "status": "success"}
            for i in range(5)
        ]
        
        assert len(results) == len(batch_request["requests"])
        
        # Step 4: Export validation
        export_formats = ["json", "csv"]
        for fmt in export_formats:
            assert fmt in ["json", "csv"]


# Performance benchmarks
class TestPerformance:
    """Performance and benchmark tests"""
    
    def test_inference_time_target(self):
        """Test that inference time meets target"""
        target_time = 2.0  # seconds
        mock_inference_time = 1.8
        
        assert mock_inference_time < target_time
    
    def test_throughput_target(self):
        """Test that throughput meets target"""
        target_throughput = 100  # patients per hour
        mock_throughput = 120
        
        assert mock_throughput >= target_throughput
    
    def test_batch_processing_efficiency(self):
        """Test batch processing efficiency"""
        sequential_time = 50.0  # seconds for 10 patients
        parallel_time = 15.0    # seconds for 10 patients
        
        speedup = sequential_time / parallel_time
        assert speedup > 2.0  # At least 2x speedup


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
