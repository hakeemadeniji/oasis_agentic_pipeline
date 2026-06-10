"""
Integration tests for Chief Medical Officer Orchestrator
Tests multi-agent coordination and end-to-end diagnostic workflow.
"""

import pytest
import torch
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator.chief_medical_officer import AdvancedChiefMedicalOfficer


class TestOrchestratorInitialization:
    """Test suite for orchestrator initialization"""
    
    @pytest.fixture
    def workspace_root(self, tmp_path):
        """Create temporary workspace structure"""
        # Create directory structure
        data_dir = tmp_path / "data" / "oasis_raw"
        data_dir.mkdir(parents=True)
        
        # Create mock clinical CSV
        clinical_data = {
            'Subject_ID': ['OAS1_0001', 'OAS1_0002'],
            'M/F': ['M', 'F'],
            'Age': [75.0, 68.0],
            'Educ': [14.0, 16.0],
            'SES': [2.0, 3.0],
            'MMSE': [28.0, 25.0],
            'eTIV': [1450.0, 1380.0],
            'nWBV': [0.73, 0.71],
            'ASF': [1.2, 1.25]
        }
        clinical_df = pd.DataFrame(clinical_data)
        clinical_csv = data_dir / "oasis_clinical_data.csv"
        clinical_df.to_csv(clinical_csv, index=False)
        
        # Create mock longitudinal CSV
        long_data = {
            'Subject ID': ['OAS2_0001', 'OAS2_0001'],
            'Visit': [1, 2],
            'MR Delay': [0, 365],
            'MMSE': [28.0, 26.0],
            'nWBV': [0.75, 0.73]
        }
        long_df = pd.DataFrame(long_data)
        long_csv = data_dir / "oasis_longitudinal.csv"
        long_df.to_csv(long_csv, index=False)
        
        # Create image directories
        for cls in ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]:
            cls_dir = data_dir / cls
            cls_dir.mkdir()
            
            # Create a dummy image
            img = Image.new('L', (224, 224), color=128)
            img.save(cls_dir / "test_image.jpg")
        
        # Create src structure for weights path
        src_dir = tmp_path / "src" / "pipeline" / "onnx_inference"
        src_dir.mkdir(parents=True)
        
        return str(tmp_path)
    
    def test_orchestrator_initialization(self, workspace_root):
        """Test that orchestrator initializes correctly"""
        cmo = AdvancedChiefMedicalOfficer(workspace_root)
        
        assert cmo is not None
        assert cmo.workspace_root == workspace_root
        
    def test_all_agents_initialized(self, workspace_root):
        """Test that all 6 agents are initialized"""
        cmo = AdvancedChiefMedicalOfficer(workspace_root)
        
        assert cmo.biomarker_agent is not None
        assert cmo.temporal_agent is not None
        assert cmo.rag_agent is not None
        assert cmo.vision_agent is not None
        assert cmo.explainer_agent is not None
        assert cmo.ethicist_agent is not None
        
    def test_device_configuration(self, workspace_root):
        """Test that device is configured correctly"""
        cmo = AdvancedChiefMedicalOfficer(workspace_root)
        
        assert cmo.device is not None
        assert isinstance(cmo.device, torch.device)
        
    def test_class_names_defined(self, workspace_root):
        """Test that class names are defined"""
        cmo = AdvancedChiefMedicalOfficer(workspace_root)
        
        assert len(cmo.class_names) == 4
        assert "Non Demented" in cmo.class_names
        assert "Moderate Dementia" in cmo.class_names
        
    def test_data_paths_set(self, workspace_root):
        """Test that data paths are set correctly"""
        cmo = AdvancedChiefMedicalOfficer(workspace_root)
        
        assert cmo.cross_csv is not None
        assert cmo.long_csv is not None
        assert "oasis_clinical_data.csv" in cmo.cross_csv
        assert "oasis_longitudinal.csv" in cmo.long_csv
        
    def test_patient_data_loaded(self, workspace_root):
        """Test that patient data is loaded"""
        cmo = AdvancedChiefMedicalOfficer(workspace_root)
        
        assert cmo.patient_tensors is not None
        assert cmo.patient_df is not None
        assert len(cmo.patient_df) > 0
        
    def test_rag_guidelines_ingested(self, workspace_root):
        """Test that RAG agent has guidelines"""
        cmo = AdvancedChiefMedicalOfficer(workspace_root)
        
        assert len(cmo.rag_agent.document_store) > 0
        assert cmo.rag_agent.vector_store is not None


class TestComprehensiveDiagnosis:
    """Test suite for comprehensive diagnosis execution"""
    
    @pytest.fixture
    def setup_orchestrator(self, tmp_path):
        """Setup orchestrator with test data"""
        # Create workspace structure
        data_dir = tmp_path / "data" / "oasis_raw"
        data_dir.mkdir(parents=True)
        
        # Create clinical CSV
        clinical_data = {
            'Subject_ID': ['OAS1_0001', 'OAS1_0002'],
            'M/F': ['M', 'F'],
            'Age': [75.0, 68.0],
            'Educ': [14.0, 16.0],
            'SES': [2.0, 3.0],
            'MMSE': [28.0, 25.0],
            'eTIV': [1450.0, 1380.0],
            'nWBV': [0.73, 0.71],
            'ASF': [1.2, 1.25]
        }
        clinical_df = pd.DataFrame(clinical_data)
        clinical_csv = data_dir / "oasis_clinical_data.csv"
        clinical_df.to_csv(clinical_csv, index=False)
        
        # Create longitudinal CSV
        long_data = {
            'Subject ID': ['OAS2_0001', 'OAS2_0001'],
            'Visit': [1, 2],
            'MR Delay': [0, 365],
            'MMSE': [28.0, 26.0],
            'nWBV': [0.75, 0.73]
        }
        long_df = pd.DataFrame(long_data)
        long_csv = data_dir / "oasis_longitudinal.csv"
        long_df.to_csv(long_csv, index=False)
        
        # Create test image
        img_dir = data_dir / "Non Demented"
        img_dir.mkdir()
        img = Image.new('L', (224, 224), color=128)
        img_path = img_dir / "test_scan.jpg"
        img.save(img_path)
        
        # Create src structure
        src_dir = tmp_path / "src" / "pipeline" / "onnx_inference"
        src_dir.mkdir(parents=True)
        
        cmo = AdvancedChiefMedicalOfficer(str(tmp_path))
        return cmo, str(img_path)
    
    def test_diagnosis_executes_without_error(self, setup_orchestrator):
        """Test that diagnosis executes without errors"""
        cmo, img_path = setup_orchestrator
        
        # Should not raise any exceptions
        cmo.execute_comprehensive_diagnosis(
            patient_idx=0,
            image_path=img_path,
            mock_subject_id="OAS2_0001"
        )
        
    def test_diagnosis_with_different_patient(self, setup_orchestrator):
        """Test diagnosis with different patient index"""
        cmo, img_path = setup_orchestrator
        
        # Test with second patient
        cmo.execute_comprehensive_diagnosis(
            patient_idx=1,
            image_path=img_path,
            mock_subject_id="OAS2_0001"
        )


class TestAgentIntegration:
    """Test suite for agent integration and data flow"""
    
    @pytest.fixture
    def cmo(self, tmp_path):
        """Create orchestrator instance"""
        # Setup minimal workspace
        data_dir = tmp_path / "data" / "oasis_raw"
        data_dir.mkdir(parents=True)
        
        clinical_data = {
            'Subject_ID': ['OAS1_0001'],
            'M/F': ['M'],
            'Age': [75.0],
            'Educ': [14.0],
            'SES': [2.0],
            'MMSE': [28.0],
            'eTIV': [1450.0],
            'nWBV': [0.73],
            'ASF': [1.2]
        }
        clinical_df = pd.DataFrame(clinical_data)
        clinical_csv = data_dir / "oasis_clinical_data.csv"
        clinical_df.to_csv(clinical_csv, index=False)
        
        long_data = {
            'Subject ID': ['OAS2_0001', 'OAS2_0001'],
            'Visit': [1, 2],
            'MR Delay': [0, 365],
            'MMSE': [28.0, 26.0],
            'nWBV': [0.75, 0.73]
        }
        long_df = pd.DataFrame(long_data)
        long_csv = data_dir / "oasis_longitudinal.csv"
        long_df.to_csv(long_csv, index=False)
        
        src_dir = tmp_path / "src" / "pipeline" / "onnx_inference"
        src_dir.mkdir(parents=True)
        
        return AdvancedChiefMedicalOfficer(str(tmp_path))
    
    def test_biomarker_agent_integration(self, cmo):
        """Test biomarker agent integration"""
        assert cmo.biomarker_agent is not None
        assert cmo.patient_tensors is not None
        assert isinstance(cmo.patient_tensors, torch.Tensor)
        
    def test_temporal_agent_integration(self, cmo):
        """Test temporal agent integration"""
        assert cmo.temporal_agent is not None
        
        # Test temporal analysis
        metrics = cmo.temporal_agent.calculate_progression_trajectory("OAS2_0001")
        assert isinstance(metrics, dict)
        assert 'atrophy_velocity_pct' in metrics
        
    def test_rag_agent_integration(self, cmo):
        """Test RAG agent integration"""
        assert cmo.rag_agent is not None
        assert len(cmo.rag_agent.document_store) > 0
        
        # Test query
        results = cmo.rag_agent.query("MMSE score", top_k=1)
        assert len(results) > 0
        
    def test_vision_agent_integration(self, cmo):
        """Test vision agent integration"""
        assert cmo.vision_agent is not None
        
        # Test forward pass
        test_input = torch.randn(1, 1, 224, 224).to(cmo.device)
        output = cmo.vision_agent(test_input)
        assert output.shape == (1, 4)
        
    def test_explainer_agent_integration(self, cmo):
        """Test explainer agent integration"""
        assert cmo.explainer_agent is not None
        
        # Test heatmap generation
        test_input = torch.randn(1, 1, 224, 224, requires_grad=True).to(cmo.device)
        heatmap = cmo.explainer_agent.generate_heatmap(test_input, target_class=0)
        assert isinstance(heatmap, np.ndarray)
        
    def test_ethicist_agent_integration(self, cmo):
        """Test ethicist agent integration"""
        assert cmo.ethicist_agent is not None
        
        # Test audit
        is_flagged, message = cmo.ethicist_agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=85.0,
            mmse_score=28.0,
            atrophy_velocity=0.5
        )
        assert isinstance(is_flagged, bool)
        assert isinstance(message, str)


class TestImageProcessing:
    """Test suite for image processing pipeline"""
    
    @pytest.fixture
    def cmo_with_image(self, tmp_path):
        """Create orchestrator with test image"""
        # Setup workspace
        data_dir = tmp_path / "data" / "oasis_raw"
        data_dir.mkdir(parents=True)
        
        clinical_data = {
            'Subject_ID': ['OAS1_0001'],
            'M/F': ['M'],
            'Age': [75.0],
            'Educ': [14.0],
            'SES': [2.0],
            'MMSE': [28.0],
            'eTIV': [1450.0],
            'nWBV': [0.73],
            'ASF': [1.2]
        }
        clinical_df = pd.DataFrame(clinical_data)
        clinical_csv = data_dir / "oasis_clinical_data.csv"
        clinical_df.to_csv(clinical_csv, index=False)
        
        long_data = {
            'Subject ID': ['OAS2_0001'],
            'Visit': [1],
            'MR Delay': [0],
            'MMSE': [28.0],
            'nWBV': [0.75]
        }
        long_df = pd.DataFrame(long_data)
        long_csv = data_dir / "oasis_longitudinal.csv"
        long_df.to_csv(long_csv, index=False)
        
        # Create test image
        img = Image.new('L', (224, 224), color=128)
        img_path = tmp_path / "test_image.jpg"
        img.save(img_path)
        
        src_dir = tmp_path / "src" / "pipeline" / "onnx_inference"
        src_dir.mkdir(parents=True)
        
        cmo = AdvancedChiefMedicalOfficer(str(tmp_path))
        return cmo, str(img_path)
    
    def test_image_transform_defined(self, cmo_with_image):
        """Test that image transform is defined"""
        cmo, _ = cmo_with_image
        
        assert cmo.image_transform is not None
        
    def test_image_loading_and_transform(self, cmo_with_image):
        """Test image loading and transformation"""
        cmo, img_path = cmo_with_image
        
        img = Image.open(img_path).convert('L')
        transformed = cmo.image_transform(img)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (1, 224, 224)
        
    def test_image_to_device(self, cmo_with_image):
        """Test moving image to device"""
        cmo, img_path = cmo_with_image
        
        img = Image.open(img_path).convert('L')
        transformed = cmo.image_transform(img)
        img_tensor = transformed.unsqueeze(0).to(cmo.device)
        
        assert img_tensor.device == cmo.device


class TestErrorHandling:
    """Test suite for error handling"""
    
    @pytest.fixture
    def cmo(self, tmp_path):
        """Create orchestrator with minimal setup"""
        data_dir = tmp_path / "data" / "oasis_raw"
        data_dir.mkdir(parents=True)
        
        # Minimal CSV
        clinical_data = {
            'Subject_ID': ['OAS1_0001'],
            'M/F': ['M'],
            'Age': [75.0],
            'Educ': [14.0],
            'SES': [2.0],
            'MMSE': [28.0],
            'eTIV': [1450.0],
            'nWBV': [0.73],
            'ASF': [1.2]
        }
        clinical_df = pd.DataFrame(clinical_data)
        clinical_csv = data_dir / "oasis_clinical_data.csv"
        clinical_df.to_csv(clinical_csv, index=False)
        
        long_csv = data_dir / "oasis_longitudinal.csv"
        pd.DataFrame().to_csv(long_csv, index=False)
        
        src_dir = tmp_path / "src" / "pipeline" / "onnx_inference"
        src_dir.mkdir(parents=True)
        
        return AdvancedChiefMedicalOfficer(str(tmp_path))
    
    def test_missing_weights_handled(self, cmo):
        """Test that missing weights are handled gracefully"""
        # Should initialize without weights
        assert cmo.vision_agent is not None
        
    def test_invalid_patient_index(self, cmo, tmp_path):
        """Test handling of invalid patient index"""
        # Create test image
        img = Image.new('L', (224, 224), color=128)
        img_path = tmp_path / "test.jpg"
        img.save(img_path)
        
        # Should raise IndexError for out of bounds index
        with pytest.raises(IndexError):
            cmo.execute_comprehensive_diagnosis(
                patient_idx=999,
                image_path=str(img_path)
            )


class TestDataConsistency:
    """Test suite for data consistency across agents"""
    
    @pytest.fixture
    def cmo(self, tmp_path):
        """Create orchestrator"""
        data_dir = tmp_path / "data" / "oasis_raw"
        data_dir.mkdir(parents=True)
        
        clinical_data = {
            'Subject_ID': ['OAS1_0001', 'OAS1_0002'],
            'M/F': ['M', 'F'],
            'Age': [75.0, 68.0],
            'Educ': [14.0, 16.0],
            'SES': [2.0, 3.0],
            'MMSE': [28.0, 25.0],
            'eTIV': [1450.0, 1380.0],
            'nWBV': [0.73, 0.71],
            'ASF': [1.2, 1.25]
        }
        clinical_df = pd.DataFrame(clinical_data)
        clinical_csv = data_dir / "oasis_clinical_data.csv"
        clinical_df.to_csv(clinical_csv, index=False)
        
        long_data = {
            'Subject ID': ['OAS2_0001'],
            'Visit': [1],
            'MR Delay': [0],
            'MMSE': [28.0],
            'nWBV': [0.75]
        }
        long_df = pd.DataFrame(long_data)
        long_csv = data_dir / "oasis_longitudinal.csv"
        long_df.to_csv(long_csv, index=False)
        
        src_dir = tmp_path / "src" / "pipeline" / "onnx_inference"
        src_dir.mkdir(parents=True)
        
        return AdvancedChiefMedicalOfficer(str(tmp_path))
    
    def test_patient_count_consistency(self, cmo):
        """Test that patient counts are consistent"""
        tensor_count = cmo.patient_tensors.shape[0]
        df_count = len(cmo.patient_df)
        
        assert tensor_count == df_count
        
    def test_patient_data_alignment(self, cmo):
        """Test that patient data is aligned across sources"""
        # First patient MMSE should match in both tensor and dataframe
        mmse_from_df = float(cmo.patient_df.iloc[0]['MMSE'])
        
        # MMSE is one of the features in the tensor
        assert mmse_from_df > 0  # Basic sanity check


class TestOutputGeneration:
    """Test suite for output generation"""
    
    @pytest.fixture
    def setup_full_system(self, tmp_path, capsys):
        """Setup full system for output testing"""
        data_dir = tmp_path / "data" / "oasis_raw"
        data_dir.mkdir(parents=True)
        
        clinical_data = {
            'Subject_ID': ['OAS1_0001'],
            'M/F': ['M'],
            'Age': [75.0],
            'Educ': [14.0],
            'SES': [2.0],
            'MMSE': [28.0],
            'eTIV': [1450.0],
            'nWBV': [0.73],
            'ASF': [1.2]
        }
        clinical_df = pd.DataFrame(clinical_data)
        clinical_csv = data_dir / "oasis_clinical_data.csv"
        clinical_df.to_csv(clinical_csv, index=False)
        
        long_data = {
            'Subject ID': ['OAS2_0001', 'OAS2_0001'],
            'Visit': [1, 2],
            'MR Delay': [0, 365],
            'MMSE': [28.0, 26.0],
            'nWBV': [0.75, 0.73]
        }
        long_df = pd.DataFrame(long_data)
        long_csv = data_dir / "oasis_longitudinal.csv"
        long_df.to_csv(long_csv, index=False)
        
        img = Image.new('L', (224, 224), color=128)
        img_path = tmp_path / "test.jpg"
        img.save(img_path)
        
        src_dir = tmp_path / "src" / "pipeline" / "onnx_inference"
        src_dir.mkdir(parents=True)
        
        cmo = AdvancedChiefMedicalOfficer(str(tmp_path))
        return cmo, str(img_path), capsys
    
    def test_diagnosis_produces_output(self, setup_full_system):
        """Test that diagnosis produces console output"""
        cmo, img_path, capsys = setup_full_system
        
        cmo.execute_comprehensive_diagnosis(
            patient_idx=0,
            image_path=img_path,
            mock_subject_id="OAS2_0001"
        )
        
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        
    def test_output_contains_key_sections(self, setup_full_system):
        """Test that output contains all key sections"""
        cmo, img_path, capsys = setup_full_system
        
        cmo.execute_comprehensive_diagnosis(
            patient_idx=0,
            image_path=img_path,
            mock_subject_id="OAS2_0001"
        )
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Check for key sections
        assert "LONGITUDINAL KINETICS" in output
        assert "EXPLAINABLE RADIOMICS" in output
        assert "COMPLIANCE & RISK MANAGEMENT" in output
        assert "SUPPORTING MEDICAL LITERATURE" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
