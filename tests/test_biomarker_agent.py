import pytest
import torch
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.biomarker.biomarker_agent import ClinicalBiomarkerAgent


class TestClinicalBiomarkerAgent:
    """Test suite for Biomarker Agent (Agent 2)"""
    
    @pytest.fixture
    def agent(self):
        """Create a biomarker agent instance"""
        return ClinicalBiomarkerAgent()
    
    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file for testing"""
        data = {
            'Subject_ID': ['OAS1_0001', 'OAS1_0002', 'OAS1_0003'],
            'M/F': ['M', 'F', 'M'],
            'Age': [75.0, 68.0, 82.0],
            'Educ': [14.0, 16.0, 12.0],
            'SES': [2.0, 3.0, 1.0],
            'MMSE': [28.0, 25.0, 22.0],
            'eTIV': [1450.0, 1380.0, 1520.0],
            'nWBV': [0.73, 0.71, 0.68],
            'ASF': [1.2, 1.25, 1.15]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_clinical.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent is not None
        assert agent.scaler is not None
        assert not agent.features_fitted
        
    def test_ingest_and_process(self, agent, sample_csv):
        """Test data ingestion and processing"""
        tensors, df = agent.ingest_and_process(sample_csv)
        
        assert isinstance(tensors, torch.Tensor)
        assert isinstance(df, pd.DataFrame)
        assert tensors.shape[0] == 3  # 3 patients
        assert tensors.shape[1] == 8  # 7 continuous + 1 categorical
        
    def test_tensor_dtype(self, agent, sample_csv):
        """Test output tensor has correct dtype"""
        tensors, df = agent.ingest_and_process(sample_csv)
        assert tensors.dtype == torch.float32
        
    def test_normalization(self, agent, sample_csv):
        """Test features are normalized"""
        tensors, df = agent.ingest_and_process(sample_csv)
        
        # Normalized features should have mean ~0 and std ~1
        # (except for the binary gender feature)
        continuous_features = tensors[:, :-1]  # Exclude gender
        mean = continuous_features.mean(dim=0)
        std = continuous_features.std(dim=0)
        
        # Check mean is close to 0 (within tolerance)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1.0)
        
    def test_gender_encoding(self, agent, sample_csv):
        """Test gender is correctly encoded"""
        tensors, df = agent.ingest_and_process(sample_csv)
        
        # Last column should be gender (0 or 1)
        gender_col = tensors[:, -1]
        assert torch.all((gender_col == 0) | (gender_col == 1))
        
    def test_missing_value_imputation(self, agent, tmp_path):
        """Test missing values are imputed"""
        data = {
            'Subject_ID': ['OAS1_0001', 'OAS1_0002', 'OAS1_0003'],
            'M/F': ['M', 'F', 'M'],
            'Age': [75.0, np.nan, 82.0],
            'Educ': [14.0, 16.0, 12.0],
            'SES': [2.0, 3.0, np.nan],
            'MMSE': [28.0, 25.0, 22.0],
            'eTIV': [1450.0, 1380.0, 1520.0],
            'nWBV': [0.73, 0.71, 0.68],
            'ASF': [1.2, 1.25, 1.15]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_missing.csv"
        df.to_csv(csv_path, index=False)
        
        tensors, df_out = agent.ingest_and_process(str(csv_path))
        
        # Check no NaN values in output
        assert not torch.isnan(tensors).any()
        
    def test_dataframe_preservation(self, agent, sample_csv):
        """Test human-readable dataframe is preserved"""
        tensors, df = agent.ingest_and_process(sample_csv)
        
        # Check original columns are preserved
        assert 'Subject_ID' in df.columns
        assert 'Age' in df.columns
        assert 'MMSE' in df.columns
        
        # Check values are not normalized in dataframe
        assert df['Age'].mean() > 10  # Should be actual age, not normalized
        
    def test_scaler_fitting(self, agent, sample_csv):
        """Test scaler is fitted on first call"""
        assert not agent.features_fitted
        
        tensors, df = agent.ingest_and_process(sample_csv)
        assert agent.features_fitted
        
    def test_consistent_scaling(self, agent, sample_csv):
        """Test scaling is consistent across multiple calls"""
        tensors1, _ = agent.ingest_and_process(sample_csv)
        tensors2, _ = agent.ingest_and_process(sample_csv)
        
        assert torch.allclose(tensors1, tensors2, atol=1e-6)
        
    def test_synthetic_data_generation(self, agent, tmp_path):
        """Test synthetic data generation when CSV missing"""
        fake_path = tmp_path / "nonexistent.csv"
        tensors, df = agent.ingest_and_process(str(fake_path))
        
        # Should generate 100 synthetic patients
        assert tensors.shape[0] == 100
        assert df.shape[0] == 100
        
        # Check CSV was created
        assert fake_path.exists()


class TestFeatureValidation:
    """Test suite for feature validation"""
    
    @pytest.fixture
    def agent(self):
        return ClinicalBiomarkerAgent()
    
    def test_age_range(self, agent, tmp_path):
        """Test age values are in valid range"""
        data = {
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
        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_age.csv"
        df.to_csv(csv_path, index=False)
        
        tensors, df_out = agent.ingest_and_process(str(csv_path))
        
        # Original age should be preserved in dataframe
        assert 60 <= df_out['Age'].iloc[0] <= 100
        
    def test_mmse_range(self, agent, tmp_path):
        """Test MMSE values are in valid range"""
        data = {
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
        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_mmse.csv"
        df.to_csv(csv_path, index=False)
        
        tensors, df_out = agent.ingest_and_process(str(csv_path))
        
        # MMSE should be between 0 and 30
        assert 0 <= df_out['MMSE'].iloc[0] <= 30


class TestEdgeCases:
    """Test suite for edge cases"""
    
    @pytest.fixture
    def agent(self):
        return ClinicalBiomarkerAgent()
    
    def test_single_patient(self, agent, tmp_path):
        """Test processing single patient"""
        data = {
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
        df = pd.DataFrame(data)
        csv_path = tmp_path / "single_patient.csv"
        df.to_csv(csv_path, index=False)
        
        tensors, df_out = agent.ingest_and_process(str(csv_path))
        
        assert tensors.shape[0] == 1
        assert df_out.shape[0] == 1
        
    def test_large_dataset(self, agent, tmp_path):
        """Test processing large dataset"""
        n_patients = 1000
        data = {
            'Subject_ID': [f'OAS1_{i:04d}' for i in range(n_patients)],
            'M/F': np.random.choice(['M', 'F'], n_patients),
            'Age': np.random.uniform(60, 95, n_patients),
            'Educ': np.random.uniform(6, 20, n_patients),
            'SES': np.random.choice([1, 2, 3, 4, 5], n_patients),
            'MMSE': np.random.uniform(0, 30, n_patients),
            'eTIV': np.random.uniform(1200, 1700, n_patients),
            'nWBV': np.random.uniform(0.6, 0.8, n_patients),
            'ASF': np.random.uniform(1.0, 1.4, n_patients)
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "large_dataset.csv"
        df.to_csv(csv_path, index=False)
        
        tensors, df_out = agent.ingest_and_process(str(csv_path))
        
        assert tensors.shape[0] == n_patients
        assert not torch.isnan(tensors).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
