"""
Unit tests for Temporal Analyst Agent (Agent 5) - Longitudinal Progression
Tests progression trajectory calculation and temporal metrics.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.biomarker.temporal_analyst import TemporalAnalystAgent


class TestTemporalAnalystInitialization:
    """Test suite for Temporal Analyst Agent initialization"""
    
    @pytest.fixture
    def sample_longitudinal_csv(self, tmp_path):
        """Create a sample longitudinal CSV file"""
        data = {
            'Subject ID': ['OAS2_0001', 'OAS2_0001', 'OAS2_0001', 'OAS2_0002', 'OAS2_0002'],
            'Visit': [1, 2, 3, 1, 2],
            'MR Delay': [0, 365, 730, 0, 400],
            'MMSE': [28.0, 26.0, 24.0, 30.0, 29.0],
            'nWBV': [0.75, 0.73, 0.71, 0.78, 0.77]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "longitudinal_test.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)
    
    def test_initialization_with_valid_csv(self, sample_longitudinal_csv):
        """Test agent initializes correctly with valid CSV"""
        agent = TemporalAnalystAgent(sample_longitudinal_csv)
        
        assert agent is not None
        assert not agent.df.empty
        assert len(agent.df) == 5
        
    def test_initialization_with_missing_csv(self, tmp_path):
        """Test agent handles missing CSV gracefully"""
        fake_path = str(tmp_path / "nonexistent.csv")
        agent = TemporalAnalystAgent(fake_path)
        
        assert agent is not None
        assert agent.df.empty
        
    def test_csv_path_stored(self, sample_longitudinal_csv):
        """Test that CSV path is stored correctly"""
        agent = TemporalAnalystAgent(sample_longitudinal_csv)
        
        assert agent.csv_path == sample_longitudinal_csv
        
    def test_dataframe_columns(self, sample_longitudinal_csv):
        """Test that required columns are present"""
        agent = TemporalAnalystAgent(sample_longitudinal_csv)
        
        required_columns = ['Subject ID', 'Visit', 'MR Delay', 'MMSE', 'nWBV']
        for col in required_columns:
            assert col in agent.df.columns


class TestProgressionTrajectoryCalculation:
    """Test suite for progression trajectory calculation"""
    
    @pytest.fixture
    def agent_with_data(self, tmp_path):
        """Create agent with sample longitudinal data"""
        data = {
            'Subject ID': ['OAS2_0001', 'OAS2_0001', 'OAS2_0001'],
            'Visit': [1, 2, 3],
            'MR Delay': [0, 365, 730],
            'MMSE': [28.0, 26.0, 24.0],
            'nWBV': [0.75, 0.73, 0.71]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        return TemporalAnalystAgent(str(csv_path))
    
    def test_calculate_trajectory_returns_dict(self, agent_with_data):
        """Test that trajectory calculation returns dictionary"""
        result = agent_with_data.calculate_progression_trajectory("OAS2_0001")
        
        assert isinstance(result, dict)
        
    def test_trajectory_required_keys(self, agent_with_data):
        """Test that result contains required keys"""
        result = agent_with_data.calculate_progression_trajectory("OAS2_0001")
        
        required_keys = ['visits_tracked', 'years_monitored', 'atrophy_velocity_pct', 
                        'mmse_drift', 'clinical_trend']
        for key in required_keys:
            assert key in result
            
    def test_visits_tracked_count(self, agent_with_data):
        """Test that visits are counted correctly"""
        result = agent_with_data.calculate_progression_trajectory("OAS2_0001")
        
        assert result['visits_tracked'] == 3
        
    def test_years_monitored_calculation(self, agent_with_data):
        """Test that years monitored is calculated correctly"""
        result = agent_with_data.calculate_progression_trajectory("OAS2_0001")
        
        # 730 days = 2 years
        assert abs(result['years_monitored'] - 2.0) < 0.1
        
    def test_atrophy_velocity_calculation(self, agent_with_data):
        """Test atrophy velocity calculation"""
        result = agent_with_data.calculate_progression_trajectory("OAS2_0001")
        
        # nWBV: 0.75 -> 0.71 over 2 years
        # Loss = 0.04, Velocity = (0.04/0.75)/2 * 100 = 2.67% per year
        expected_velocity = (0.04 / 0.75) / 2.0 * 100
        assert abs(result['atrophy_velocity_pct'] - expected_velocity) < 0.1
        
    def test_mmse_drift_calculation(self, agent_with_data):
        """Test MMSE drift calculation"""
        result = agent_with_data.calculate_progression_trajectory("OAS2_0001")
        
        # MMSE: 28 -> 24, drift = -4
        assert result['mmse_drift'] == -4
        
    def test_clinical_trend_classification(self, agent_with_data):
        """Test clinical trend is classified"""
        result = agent_with_data.calculate_progression_trajectory("OAS2_0001")
        
        assert 'clinical_trend' in result
        assert isinstance(result['clinical_trend'], str)
        assert len(result['clinical_trend']) > 0


class TestClinicalTrendClassification:
    """Test suite for clinical trend classification logic"""
    
    @pytest.fixture
    def create_agent_with_trend(self, tmp_path):
        """Factory to create agent with specific trend data"""
        def _create(atrophy_rate, mmse_change):
            # Calculate nWBV values to achieve desired atrophy rate over 2 years
            baseline_nwbv = 0.75
            years = 2.0
            total_loss = (atrophy_rate / 100) * baseline_nwbv * years
            final_nwbv = baseline_nwbv - total_loss
            
            # Calculate MMSE values
            baseline_mmse = 28.0
            final_mmse = baseline_mmse + mmse_change
            
            data = {
                'Subject ID': ['TEST_001', 'TEST_001'],
                'Visit': [1, 2],
                'MR Delay': [0, 730],
                'MMSE': [baseline_mmse, final_mmse],
                'nWBV': [baseline_nwbv, final_nwbv]
            }
            df = pd.DataFrame(data)
            csv_path = tmp_path / f"trend_test_{atrophy_rate}_{mmse_change}.csv"
            df.to_csv(csv_path, index=False)
            return TemporalAnalystAgent(str(csv_path))
        return _create
    
    def test_aggressive_decline_high_atrophy(self, create_agent_with_trend):
        """Test aggressive decline classification with high atrophy"""
        agent = create_agent_with_trend(atrophy_rate=2.0, mmse_change=-2)
        result = agent.calculate_progression_trajectory("TEST_001")
        
        assert "Aggressive" in result['clinical_trend']
        
    def test_aggressive_decline_mmse_drop(self, create_agent_with_trend):
        """Test aggressive decline classification with MMSE drop"""
        agent = create_agent_with_trend(atrophy_rate=1.0, mmse_change=-4)
        result = agent.calculate_progression_trajectory("TEST_001")
        
        assert "Aggressive" in result['clinical_trend']
        
    def test_typical_age_related(self, create_agent_with_trend):
        """Test typical age-related classification"""
        agent = create_agent_with_trend(atrophy_rate=0.8, mmse_change=-1)
        result = agent.calculate_progression_trajectory("TEST_001")
        
        assert "Typical" in result['clinical_trend']
        
    def test_stable_maintenance(self, create_agent_with_trend):
        """Test stable maintenance classification"""
        agent = create_agent_with_trend(atrophy_rate=0.3, mmse_change=0)
        result = agent.calculate_progression_trajectory("TEST_001")
        
        assert "Stable" in result['clinical_trend']


class TestEdgeCases:
    """Test suite for edge cases and error handling"""
    
    @pytest.fixture
    def empty_agent(self, tmp_path):
        """Create agent with no data"""
        fake_path = str(tmp_path / "nonexistent.csv")
        return TemporalAnalystAgent(fake_path)
    
    def test_no_database_loaded(self, empty_agent):
        """Test handling when no database is loaded"""
        result = empty_agent.calculate_progression_trajectory("OAS2_0001")
        
        assert 'status' in result
        assert result['atrophy_velocity_pct'] == 0.0
        assert result['mmse_drift'] == 0.0
        
    def test_single_visit_patient(self, tmp_path):
        """Test patient with only one visit"""
        data = {
            'Subject ID': ['OAS2_0001'],
            'Visit': [1],
            'MR Delay': [0],
            'MMSE': [28.0],
            'nWBV': [0.75]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "single_visit.csv"
        df.to_csv(csv_path, index=False)
        
        agent = TemporalAnalystAgent(str(csv_path))
        result = agent.calculate_progression_trajectory("OAS2_0001")
        
        assert result['visits_tracked'] == 1
        assert result['atrophy_velocity_pct'] == 0.0
        assert result['mmse_drift'] == 0.0
        assert "Baseline Only" in result['clinical_trend']
        
    def test_nonexistent_patient(self, tmp_path):
        """Test querying patient not in database"""
        data = {
            'Subject ID': ['OAS2_0001'],
            'Visit': [1],
            'MR Delay': [0],
            'MMSE': [28.0],
            'nWBV': [0.75]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        agent = TemporalAnalystAgent(str(csv_path))
        result = agent.calculate_progression_trajectory("NONEXISTENT_ID")
        
        assert result['visits_tracked'] == 0
        assert result['atrophy_velocity_pct'] == 0.0
        
    def test_zero_time_elapsed(self, tmp_path):
        """Test handling when all visits have same MR Delay"""
        data = {
            'Subject ID': ['OAS2_0001', 'OAS2_0001'],
            'Visit': [1, 2],
            'MR Delay': [0, 0],  # Same delay
            'MMSE': [28.0, 27.0],
            'nWBV': [0.75, 0.74]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "zero_time.csv"
        df.to_csv(csv_path, index=False)
        
        agent = TemporalAnalystAgent(str(csv_path))
        result = agent.calculate_progression_trajectory("OAS2_0001")
        
        # Should handle division by zero gracefully
        assert isinstance(result['atrophy_velocity_pct'], (int, float))
        assert not np.isnan(result['atrophy_velocity_pct'])
        assert not np.isinf(result['atrophy_velocity_pct'])


class TestMultiplePatients:
    """Test suite for handling multiple patients"""
    
    @pytest.fixture
    def multi_patient_agent(self, tmp_path):
        """Create agent with multiple patients"""
        data = {
            'Subject ID': ['OAS2_0001', 'OAS2_0001', 'OAS2_0002', 'OAS2_0002', 'OAS2_0003'],
            'Visit': [1, 2, 1, 2, 1],
            'MR Delay': [0, 365, 0, 730, 0],
            'MMSE': [28.0, 26.0, 30.0, 28.0, 25.0],
            'nWBV': [0.75, 0.73, 0.78, 0.76, 0.70]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "multi_patient.csv"
        df.to_csv(csv_path, index=False)
        return TemporalAnalystAgent(str(csv_path))
    
    def test_different_patients_different_results(self, multi_patient_agent):
        """Test that different patients get different results"""
        result1 = multi_patient_agent.calculate_progression_trajectory("OAS2_0001")
        result2 = multi_patient_agent.calculate_progression_trajectory("OAS2_0002")
        
        # Results should differ
        assert result1['visits_tracked'] != result2['visits_tracked'] or \
               result1['atrophy_velocity_pct'] != result2['atrophy_velocity_pct']
               
    def test_patient_isolation(self, multi_patient_agent):
        """Test that patient data is properly isolated"""
        result = multi_patient_agent.calculate_progression_trajectory("OAS2_0001")
        
        # Should only count visits for OAS2_0001
        assert result['visits_tracked'] == 2


class TestDataValidation:
    """Test suite for data validation"""
    
    def test_negative_atrophy_velocity(self, tmp_path):
        """Test handling of brain volume increase (unusual but possible)"""
        data = {
            'Subject ID': ['OAS2_0001', 'OAS2_0001'],
            'Visit': [1, 2],
            'MR Delay': [0, 365],
            'MMSE': [28.0, 29.0],
            'nWBV': [0.70, 0.75]  # Volume increased
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "negative_atrophy.csv"
        df.to_csv(csv_path, index=False)
        
        agent = TemporalAnalystAgent(str(csv_path))
        result = agent.calculate_progression_trajectory("OAS2_0001")
        
        # Negative atrophy velocity (brain volume increased)
        assert result['atrophy_velocity_pct'] < 0
        
    def test_positive_mmse_drift(self, tmp_path):
        """Test handling of MMSE improvement"""
        data = {
            'Subject ID': ['OAS2_0001', 'OAS2_0001'],
            'Visit': [1, 2],
            'MR Delay': [0, 365],
            'MMSE': [24.0, 28.0],  # Improved
            'nWBV': [0.75, 0.74]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "mmse_improvement.csv"
        df.to_csv(csv_path, index=False)
        
        agent = TemporalAnalystAgent(str(csv_path))
        result = agent.calculate_progression_trajectory("OAS2_0001")
        
        # Positive MMSE drift (improvement)
        assert result['mmse_drift'] > 0
        
    def test_extreme_values(self, tmp_path):
        """Test handling of extreme but valid values"""
        data = {
            'Subject ID': ['OAS2_0001', 'OAS2_0001'],
            'Visit': [1, 2],
            'MR Delay': [0, 3650],  # 10 years
            'MMSE': [30.0, 0.0],  # Severe decline
            'nWBV': [0.80, 0.50]  # Severe atrophy
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "extreme_values.csv"
        df.to_csv(csv_path, index=False)
        
        agent = TemporalAnalystAgent(str(csv_path))
        result = agent.calculate_progression_trajectory("OAS2_0001")
        
        assert isinstance(result['atrophy_velocity_pct'], (int, float))
        assert isinstance(result['mmse_drift'], (int, float))
        assert not np.isnan(result['atrophy_velocity_pct'])


class TestOutputFormatting:
    """Test suite for output formatting"""
    
    @pytest.fixture
    def agent_with_data(self, tmp_path):
        """Create agent with sample data"""
        data = {
            'Subject ID': ['OAS2_0001', 'OAS2_0001'],
            'Visit': [1, 2],
            'MR Delay': [0, 365],
            'MMSE': [28.0, 26.0],
            'nWBV': [0.75, 0.73]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "format_test.csv"
        df.to_csv(csv_path, index=False)
        return TemporalAnalystAgent(str(csv_path))
    
    def test_numeric_precision(self, agent_with_data):
        """Test that numeric values have appropriate precision"""
        result = agent_with_data.calculate_progression_trajectory("OAS2_0001")
        
        # Years should be rounded to 2 decimals
        assert isinstance(result['years_monitored'], float)
        
        # Atrophy velocity should be rounded to 3 decimals
        assert isinstance(result['atrophy_velocity_pct'], float)
        
        # MMSE drift should be integer
        assert isinstance(result['mmse_drift'], int)
        
    def test_visits_tracked_is_integer(self, agent_with_data):
        """Test that visits tracked is an integer"""
        result = agent_with_data.calculate_progression_trajectory("OAS2_0001")
        
        assert isinstance(result['visits_tracked'], int)
        
    def test_clinical_trend_is_string(self, agent_with_data):
        """Test that clinical trend is a string"""
        result = agent_with_data.calculate_progression_trajectory("OAS2_0001")
        
        assert isinstance(result['clinical_trend'], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
