"""
Unit tests for Medical Ethicist Agent (Agent 6) - Compliance Guardrails
Tests diagnostic proposal auditing and clinical safety rules.
"""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator.ethicist_agent import MedicalEthicistAgent


class TestMedicalEthicistInitialization:
    """Test suite for Medical Ethicist Agent initialization"""
    
    def test_default_initialization(self):
        """Test agent initializes with default confidence floor"""
        agent = MedicalEthicistAgent()
        
        assert agent is not None
        assert agent.confidence_floor == 65.0
        
    def test_custom_confidence_floor(self):
        """Test agent initializes with custom confidence floor"""
        agent = MedicalEthicistAgent(confidence_floor=70.0)
        
        assert agent.confidence_floor == 70.0
        
    def test_low_confidence_floor(self):
        """Test agent with low confidence floor"""
        agent = MedicalEthicistAgent(confidence_floor=50.0)
        
        assert agent.confidence_floor == 50.0
        
    def test_high_confidence_floor(self):
        """Test agent with high confidence floor"""
        agent = MedicalEthicistAgent(confidence_floor=90.0)
        
        assert agent.confidence_floor == 90.0


class TestAuditDiagnosticProposal:
    """Test suite for diagnostic proposal auditing"""
    
    @pytest.fixture
    def agent(self):
        """Create ethicist agent with default settings"""
        return MedicalEthicistAgent()
    
    def test_audit_returns_tuple(self, agent):
        """Test that audit returns a tuple"""
        result = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=85.0,
            mmse_score=28.0,
            atrophy_velocity=0.5
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
    def test_audit_tuple_format(self, agent):
        """Test that audit returns (bool, str) format"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=85.0,
            mmse_score=28.0,
            atrophy_velocity=0.5
        )
        
        assert isinstance(is_flagged, bool)
        assert isinstance(message, str)
        
    def test_valid_diagnosis_approved(self, agent):
        """Test that valid diagnosis is approved"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=85.0,
            mmse_score=28.0,
            atrophy_velocity=0.5
        )
        
        assert not is_flagged
        assert "VERIFIED" in message


class TestConfidenceThresholdRule:
    """Test suite for Rule 1: High Uncertainty Attenuation"""
    
    @pytest.fixture
    def agent(self):
        """Create ethicist agent"""
        return MedicalEthicistAgent(confidence_floor=65.0)
    
    def test_below_threshold_rejected(self, agent):
        """Test that confidence below threshold is rejected"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=60.0,  # Below 65%
            mmse_score=28.0,
            atrophy_velocity=0.5
        )
        
        assert is_flagged
        assert "REJECTED" in message
        assert "confidence" in message.lower()
        
    def test_at_threshold_approved(self, agent):
        """Test that confidence at threshold is approved"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=65.0,  # Exactly at threshold
            mmse_score=28.0,
            atrophy_velocity=0.5
        )
        
        assert not is_flagged
        
    def test_above_threshold_approved(self, agent):
        """Test that confidence above threshold is approved"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=85.0,  # Above threshold
            mmse_score=28.0,
            atrophy_velocity=0.5
        )
        
        assert not is_flagged
        
    def test_very_low_confidence_rejected(self, agent):
        """Test that very low confidence is rejected"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=30.0,
            mmse_score=28.0,
            atrophy_velocity=0.5
        )
        
        assert is_flagged
        assert "Sub-threshold" in message
        
    def test_confidence_floor_in_message(self, agent):
        """Test that confidence floor is mentioned in rejection message"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=50.0,
            mmse_score=28.0,
            atrophy_velocity=0.5
        )
        
        assert "65" in message  # Confidence floor value


class TestCognitiveVsSpatialContradiction:
    """Test suite for Rule 2: Cognitive vs Spatial Contradiction Guardrail"""
    
    @pytest.fixture
    def agent(self):
        """Create ethicist agent"""
        return MedicalEthicistAgent()
    
    def test_high_mmse_with_moderate_dementia_rejected(self, agent):
        """Test that high MMSE with Moderate Dementia is rejected"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Moderate Dementia",
            confidence=85.0,
            mmse_score=29.0,  # Near perfect cognition
            atrophy_velocity=0.5
        )
        
        assert is_flagged
        assert "REJECTED" in message
        assert "Cross-Modal" in message
        
    def test_high_mmse_with_mild_dementia_rejected(self, agent):
        """Test that high MMSE with Mild Dementia is rejected"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Mild Dementia",
            confidence=85.0,
            mmse_score=28.0,
            atrophy_velocity=0.5
        )
        
        assert is_flagged
        assert "Cross-Modal" in message
        
    def test_high_mmse_with_very_mild_approved(self, agent):
        """Test that high MMSE with Very Mild Dementia is approved"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Very mild Dementia",
            confidence=85.0,
            mmse_score=27.5,
            atrophy_velocity=0.5
        )
        
        # Should not trigger Rule 2 (MMSE >= 27 with Mild/Moderate)
        assert not is_flagged
        
    def test_high_mmse_with_non_demented_approved(self, agent):
        """Test that high MMSE with Non Demented is approved"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=85.0,
            mmse_score=29.0,
            atrophy_velocity=0.5
        )
        
        assert not is_flagged
        
    def test_mmse_boundary_27(self, agent):
        """Test MMSE boundary at 27.0"""
        # At 27.0, should trigger rule
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Moderate Dementia",
            confidence=85.0,
            mmse_score=27.0,
            atrophy_velocity=0.5
        )
        
        assert is_flagged
        
    def test_mmse_just_below_boundary(self, agent):
        """Test MMSE just below 27.0"""
        # Below 27.0, should not trigger rule
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Moderate Dementia",
            confidence=85.0,
            mmse_score=26.9,
            atrophy_velocity=0.5
        )
        
        # Should pass Rule 2, but may fail other rules
        # Check that it's not rejected for Cross-Modal variance
        if is_flagged:
            assert "Cross-Modal" not in message


class TestSilentStructuralDegradation:
    """Test suite for Rule 3: Asymptomatic Silent Structural Degradation Alert"""
    
    @pytest.fixture
    def agent(self):
        """Create ethicist agent"""
        return MedicalEthicistAgent()
    
    def test_high_atrophy_with_normal_cognition_warning(self, agent):
        """Test that high atrophy with normal cognition triggers warning"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=85.0,
            mmse_score=29.0,
            atrophy_velocity=2.5  # High atrophy
        )
        
        assert not is_flagged  # Approved but with warning
        assert "WARNING" in message
        assert "structural brain tissue loss" in message.lower()
        
    def test_atrophy_boundary_2_0(self, agent):
        """Test atrophy velocity boundary at 2.0"""
        # Just above 2.0, should trigger warning
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=85.0,
            mmse_score=28.5,
            atrophy_velocity=2.1
        )
        
        assert "WARNING" in message
        
    def test_atrophy_below_threshold_no_warning(self, agent):
        """Test that atrophy below 2.0 doesn't trigger warning"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=85.0,
            mmse_score=29.0,
            atrophy_velocity=1.5
        )
        
        # Should not have warning about structural loss
        assert "WARNING" not in message or "structural brain tissue loss" not in message.lower()
        
    def test_high_atrophy_low_mmse_no_warning(self, agent):
        """Test that high atrophy with low MMSE doesn't trigger Rule 3"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Mild Dementia",
            confidence=85.0,
            mmse_score=22.0,  # Low MMSE
            atrophy_velocity=3.0
        )
        
        # Should not trigger Rule 3 (requires MMSE >= 28)
        if not is_flagged:
            assert "structural brain tissue loss" not in message.lower()
            
    def test_mmse_boundary_28(self, agent):
        """Test MMSE boundary at 28.0 for Rule 3"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=85.0,
            mmse_score=28.0,  # Exactly at boundary
            atrophy_velocity=2.5
        )
        
        assert "WARNING" in message


class TestCriticalFailureCheck:
    """Test suite for Rule 4: Critical Failure Check"""
    
    @pytest.fixture
    def agent(self):
        """Create ethicist agent"""
        return MedicalEthicistAgent()
    
    def test_low_mmse_non_demented_rejected(self, agent):
        """Test that low MMSE with Non Demented is rejected"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=85.0,
            mmse_score=10.0,  # Very low
            atrophy_velocity=0.5
        )
        
        assert is_flagged
        assert "REJECTED" in message
        assert "Type-II Error" in message
        
    def test_mmse_boundary_12(self, agent):
        """Test MMSE boundary at 12.0"""
        # At 12.0, should trigger rule
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=85.0,
            mmse_score=12.0,
            atrophy_velocity=0.5
        )
        
        assert is_flagged
        
    def test_mmse_just_above_boundary(self, agent):
        """Test MMSE just above 12.0"""
        # Above 12.0, should not trigger Rule 4
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=85.0,
            mmse_score=12.1,
            atrophy_velocity=0.5
        )
        
        # Should not be rejected for Type-II Error
        if is_flagged:
            assert "Type-II Error" not in message
            
    def test_low_mmse_with_dementia_approved(self, agent):
        """Test that low MMSE with dementia diagnosis is approved"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Moderate Dementia",
            confidence=85.0,
            mmse_score=10.0,
            atrophy_velocity=0.5
        )
        
        # Should not trigger Rule 4 (only for Non Demented)
        if is_flagged:
            assert "Type-II Error" not in message


class TestRulePriority:
    """Test suite for rule priority and interaction"""
    
    @pytest.fixture
    def agent(self):
        """Create ethicist agent"""
        return MedicalEthicistAgent()
    
    def test_confidence_rule_checked_first(self, agent):
        """Test that confidence rule is checked before others"""
        # Low confidence should be rejected even if other rules would pass
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=50.0,  # Low confidence
            mmse_score=28.0,
            atrophy_velocity=0.5
        )
        
        assert is_flagged
        assert "confidence" in message.lower()
        
    def test_multiple_violations(self, agent):
        """Test case with multiple rule violations"""
        # Low confidence AND cross-modal contradiction
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Moderate Dementia",
            confidence=50.0,  # Low confidence
            mmse_score=29.0,  # High MMSE
            atrophy_velocity=0.5
        )
        
        # Should be rejected (confidence checked first)
        assert is_flagged
        
    def test_warning_vs_rejection(self, agent):
        """Test that warnings don't override rejections"""
        # High atrophy (warning) but also cross-modal contradiction (rejection)
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Moderate Dementia",
            confidence=85.0,
            mmse_score=29.0,  # Triggers Rule 2
            atrophy_velocity=3.0  # Would trigger Rule 3
        )
        
        # Should be rejected for Rule 2, not just warned
        assert is_flagged
        assert "Cross-Modal" in message


class TestEdgeCases:
    """Test suite for edge cases"""
    
    @pytest.fixture
    def agent(self):
        """Create ethicist agent"""
        return MedicalEthicistAgent()
    
    def test_zero_confidence(self, agent):
        """Test with zero confidence"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=0.0,
            mmse_score=28.0,
            atrophy_velocity=0.5
        )
        
        assert is_flagged
        
    def test_hundred_percent_confidence(self, agent):
        """Test with 100% confidence"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=100.0,
            mmse_score=28.0,
            atrophy_velocity=0.5
        )
        
        assert not is_flagged
        
    def test_zero_mmse(self, agent):
        """Test with MMSE of 0"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Moderate Dementia",
            confidence=85.0,
            mmse_score=0.0,
            atrophy_velocity=0.5
        )
        
        # Should not trigger Rule 4 (only for Non Demented)
        if is_flagged:
            assert "Type-II Error" not in message
            
    def test_max_mmse(self, agent):
        """Test with maximum MMSE of 30"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=85.0,
            mmse_score=30.0,
            atrophy_velocity=0.5
        )
        
        assert not is_flagged
        
    def test_negative_atrophy(self, agent):
        """Test with negative atrophy velocity (brain volume increase)"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=85.0,
            mmse_score=28.0,
            atrophy_velocity=-0.5
        )
        
        # Should not trigger Rule 3 (requires > 2.0)
        assert "WARNING" not in message or "structural brain tissue loss" not in message.lower()
        
    def test_very_high_atrophy(self, agent):
        """Test with very high atrophy velocity"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Moderate Dementia",
            confidence=85.0,
            mmse_score=15.0,
            atrophy_velocity=10.0
        )
        
        # Should be approved (consistent with diagnosis)
        assert not is_flagged


class TestDiagnosticClasses:
    """Test suite for different diagnostic classes"""
    
    @pytest.fixture
    def agent(self):
        """Create ethicist agent"""
        return MedicalEthicistAgent()
    
    def test_non_demented_class(self, agent):
        """Test Non Demented classification"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=85.0,
            mmse_score=28.0,
            atrophy_velocity=0.5
        )
        
        assert not is_flagged
        
    def test_very_mild_dementia_class(self, agent):
        """Test Very mild Dementia classification"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Very mild Dementia",
            confidence=85.0,
            mmse_score=26.0,
            atrophy_velocity=1.0
        )
        
        assert not is_flagged
        
    def test_mild_dementia_class(self, agent):
        """Test Mild Dementia classification"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Mild Dementia",
            confidence=85.0,
            mmse_score=22.0,
            atrophy_velocity=1.5
        )
        
        assert not is_flagged
        
    def test_moderate_dementia_class(self, agent):
        """Test Moderate Dementia classification"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Moderate Dementia",
            confidence=85.0,
            mmse_score=15.0,
            atrophy_velocity=2.0
        )
        
        assert not is_flagged


class TestMessageContent:
    """Test suite for message content and formatting"""
    
    @pytest.fixture
    def agent(self):
        """Create ethicist agent"""
        return MedicalEthicistAgent()
    
    def test_rejection_message_format(self, agent):
        """Test that rejection messages start with REJECTED"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=50.0,
            mmse_score=28.0,
            atrophy_velocity=0.5
        )
        
        if is_flagged:
            assert message.startswith("REJECTED")
            
    def test_approval_message_format(self, agent):
        """Test that approval messages contain VERIFIED or APPROVED"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=85.0,
            mmse_score=28.0,
            atrophy_velocity=0.5
        )
        
        if not is_flagged:
            assert "VERIFIED" in message or "APPROVED" in message
            
    def test_warning_message_format(self, agent):
        """Test that warning messages contain WARNING"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=85.0,
            mmse_score=29.0,
            atrophy_velocity=2.5
        )
        
        assert "WARNING" in message
        
    def test_message_includes_values(self, agent):
        """Test that messages include relevant values"""
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Moderate Dementia",
            confidence=85.0,
            mmse_score=29.0,
            atrophy_velocity=0.5
        )
        
        # Should include MMSE value in cross-modal rejection
        if is_flagged and "Cross-Modal" in message:
            assert "29" in message or "29.0" in message


class TestCustomConfidenceFloor:
    """Test suite for custom confidence floor settings"""
    
    def test_lower_confidence_floor(self):
        """Test with lower confidence floor"""
        agent = MedicalEthicistAgent(confidence_floor=50.0)
        
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=55.0,
            mmse_score=28.0,
            atrophy_velocity=0.5
        )
        
        # Should pass with 50% floor
        assert not is_flagged
        
    def test_higher_confidence_floor(self):
        """Test with higher confidence floor"""
        agent = MedicalEthicistAgent(confidence_floor=80.0)
        
        is_flagged, message = agent.audit_diagnostic_proposal(
            predicted_class="Non Demented",
            confidence=75.0,
            mmse_score=28.0,
            atrophy_velocity=0.5
        )
        
        # Should fail with 80% floor
        assert is_flagged
        assert "confidence" in message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
