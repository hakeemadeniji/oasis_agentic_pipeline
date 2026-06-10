"""
Unit tests for Explainer Agent (Agent 4) - Radiomics Explainer
Tests Grad-CAM heatmap generation and explainability functionality.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.vision.explainer_agent import RadiomicsExplainerAgent
from agents.vision.vision_agent import AlzheimerVisionAgent


class TestRadiomicsExplainerAgent:
    """Test suite for Explainer Agent initialization"""
    
    @pytest.fixture
    def vision_model(self):
        """Create a vision model for testing"""
        return AlzheimerVisionAgent(num_classes=4)
    
    @pytest.fixture
    def explainer_agent(self, vision_model):
        """Create an explainer agent instance"""
        return RadiomicsExplainerAgent(vision_model)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor with gradient tracking"""
        return torch.randn(1, 1, 224, 224, requires_grad=True)
    
    def test_agent_initialization(self, explainer_agent):
        """Test that agent initializes correctly"""
        assert explainer_agent is not None
        assert explainer_agent.model is not None
        
    def test_model_in_eval_mode(self, explainer_agent):
        """Test that model is in evaluation mode"""
        assert not explainer_agent.model.training
        
    def test_hooks_registered(self, explainer_agent):
        """Test that forward and backward hooks are registered"""
        # Check that activations and gradients tensors exist
        assert hasattr(explainer_agent, 'activations')
        assert hasattr(explainer_agent, 'gradients')
        
    def test_target_layer_exists(self, vision_model):
        """Test that target layer (layer4) exists in model"""
        layers = dict(vision_model.named_modules())
        assert 'backbone.layer4' in layers
        
    def test_invalid_model_raises_error(self):
        """Test that invalid model raises error"""
        # Create a model without layer4
        invalid_model = nn.Sequential(nn.Linear(10, 10))
        
        with pytest.raises(AttributeError, match="Failed to locate backbone.layer4"):
            RadiomicsExplainerAgent(invalid_model)


class TestHeatmapGeneration:
    """Test suite for heatmap generation functionality"""
    
    @pytest.fixture
    def explainer_agent(self):
        """Create explainer agent with vision model"""
        model = AlzheimerVisionAgent(num_classes=4)
        return RadiomicsExplainerAgent(model)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor"""
        return torch.randn(1, 1, 224, 224, requires_grad=True)
    
    def test_generate_heatmap_returns_array(self, explainer_agent, sample_input):
        """Test that heatmap generation returns numpy array"""
        heatmap = explainer_agent.generate_heatmap(sample_input, target_class=0)
        
        assert isinstance(heatmap, np.ndarray)
        
    def test_heatmap_shape(self, explainer_agent, sample_input):
        """Test that heatmap has correct shape"""
        heatmap = explainer_agent.generate_heatmap(sample_input, target_class=0)
        
        # Heatmap should be 2D spatial map
        assert len(heatmap.shape) == 2
        assert heatmap.shape[0] > 0
        assert heatmap.shape[1] > 0
        
    def test_heatmap_value_range(self, explainer_agent, sample_input):
        """Test that heatmap values are normalized to [0, 1]"""
        heatmap = explainer_agent.generate_heatmap(sample_input, target_class=0)
        
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
        
    def test_heatmap_no_nan(self, explainer_agent, sample_input):
        """Test that heatmap doesn't contain NaN values"""
        heatmap = explainer_agent.generate_heatmap(sample_input, target_class=0)
        
        assert not np.isnan(heatmap).any()
        
    def test_heatmap_no_inf(self, explainer_agent, sample_input):
        """Test that heatmap doesn't contain infinite values"""
        heatmap = explainer_agent.generate_heatmap(sample_input, target_class=0)
        
        assert not np.isinf(heatmap).any()
        
    def test_different_target_classes(self, explainer_agent, sample_input):
        """Test heatmap generation for different target classes"""
        for target_class in range(4):  # 4 classes
            heatmap = explainer_agent.generate_heatmap(sample_input, target_class)
            
            assert isinstance(heatmap, np.ndarray)
            assert heatmap.shape[0] > 0
            
    def test_heatmap_varies_by_class(self, explainer_agent, sample_input):
        """Test that heatmaps differ for different target classes"""
        heatmap_class0 = explainer_agent.generate_heatmap(sample_input, target_class=0)
        heatmap_class1 = explainer_agent.generate_heatmap(sample_input, target_class=1)
        
        # Heatmaps should be different for different classes
        assert not np.allclose(heatmap_class0, heatmap_class1)


class TestGradientFlow:
    """Test suite for gradient flow and backpropagation"""
    
    @pytest.fixture
    def explainer_agent(self):
        """Create explainer agent"""
        model = AlzheimerVisionAgent(num_classes=4)
        return RadiomicsExplainerAgent(model)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input with gradient tracking"""
        return torch.randn(1, 1, 224, 224, requires_grad=True)
    
    def test_gradients_captured(self, explainer_agent, sample_input):
        """Test that gradients are captured during backward pass"""
        heatmap = explainer_agent.generate_heatmap(sample_input, target_class=0)
        
        # After heatmap generation, gradients should be captured
        assert explainer_agent.gradients.numel() > 0
        
    def test_activations_captured(self, explainer_agent, sample_input):
        """Test that activations are captured during forward pass"""
        heatmap = explainer_agent.generate_heatmap(sample_input, target_class=0)
        
        # After heatmap generation, activations should be captured
        assert explainer_agent.activations.numel() > 0
        
    def test_gradient_shape(self, explainer_agent, sample_input):
        """Test that gradient shape matches activation shape"""
        heatmap = explainer_agent.generate_heatmap(sample_input, target_class=0)
        
        assert explainer_agent.gradients.shape == explainer_agent.activations.shape
        
    def test_model_zero_grad(self, explainer_agent, sample_input):
        """Test that model gradients are zeroed before computation"""
        # Generate first heatmap
        heatmap1 = explainer_agent.generate_heatmap(sample_input, target_class=0)
        
        # Generate second heatmap (should zero gradients first)
        heatmap2 = explainer_agent.generate_heatmap(sample_input, target_class=1)
        
        # Both should succeed without gradient accumulation issues
        assert isinstance(heatmap1, np.ndarray)
        assert isinstance(heatmap2, np.ndarray)


class TestHeatmapNormalization:
    """Test suite for heatmap normalization"""
    
    @pytest.fixture
    def explainer_agent(self):
        """Create explainer agent"""
        model = AlzheimerVisionAgent(num_classes=4)
        return RadiomicsExplainerAgent(model)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input"""
        return torch.randn(1, 1, 224, 224, requires_grad=True)
    
    def test_min_max_normalization(self, explainer_agent, sample_input):
        """Test that heatmap uses min-max normalization"""
        heatmap = explainer_agent.generate_heatmap(sample_input, target_class=0)
        
        # After min-max normalization, min should be 0 and max should be 1
        assert np.isclose(heatmap.min(), 0.0, atol=1e-6)
        assert np.isclose(heatmap.max(), 1.0, atol=1e-6)
        
    def test_relu_applied(self, explainer_agent, sample_input):
        """Test that ReLU is applied (no negative values)"""
        heatmap = explainer_agent.generate_heatmap(sample_input, target_class=0)
        
        # After ReLU, all values should be non-negative
        assert (heatmap >= 0).all()
        
    def test_zero_division_handling(self, explainer_agent):
        """Test handling of zero division in normalization"""
        # Create input that might produce uniform activations
        uniform_input = torch.ones(1, 1, 224, 224, requires_grad=True)
        
        # Should not raise division by zero error
        heatmap = explainer_agent.generate_heatmap(uniform_input, target_class=0)
        
        assert isinstance(heatmap, np.ndarray)
        assert not np.isnan(heatmap).any()


class TestEdgeCases:
    """Test suite for edge cases"""
    
    @pytest.fixture
    def explainer_agent(self):
        """Create explainer agent"""
        model = AlzheimerVisionAgent(num_classes=4)
        return RadiomicsExplainerAgent(model)
    
    def test_batch_size_one(self, explainer_agent):
        """Test with batch size of 1"""
        input_tensor = torch.randn(1, 1, 224, 224, requires_grad=True)
        heatmap = explainer_agent.generate_heatmap(input_tensor, target_class=0)
        
        assert isinstance(heatmap, np.ndarray)
        
    def test_all_zero_input(self, explainer_agent):
        """Test with all-zero input"""
        input_tensor = torch.zeros(1, 1, 224, 224, requires_grad=True)
        heatmap = explainer_agent.generate_heatmap(input_tensor, target_class=0)
        
        assert isinstance(heatmap, np.ndarray)
        assert not np.isnan(heatmap).any()
        
    def test_all_ones_input(self, explainer_agent):
        """Test with all-ones input"""
        input_tensor = torch.ones(1, 1, 224, 224, requires_grad=True)
        heatmap = explainer_agent.generate_heatmap(input_tensor, target_class=0)
        
        assert isinstance(heatmap, np.ndarray)
        assert not np.isnan(heatmap).any()
        
    def test_extreme_values_input(self, explainer_agent):
        """Test with extreme input values"""
        input_tensor = torch.randn(1, 1, 224, 224, requires_grad=True) * 100
        heatmap = explainer_agent.generate_heatmap(input_tensor, target_class=0)
        
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
        
    def test_invalid_target_class(self, explainer_agent):
        """Test with invalid target class index"""
        input_tensor = torch.randn(1, 1, 224, 224, requires_grad=True)
        
        # Class index out of range should raise error
        with pytest.raises(IndexError):
            explainer_agent.generate_heatmap(input_tensor, target_class=10)


class TestConsistency:
    """Test suite for consistency and reproducibility"""
    
    @pytest.fixture
    def explainer_agent(self):
        """Create explainer agent"""
        model = AlzheimerVisionAgent(num_classes=4)
        return RadiomicsExplainerAgent(model)
    
    def test_deterministic_output(self, explainer_agent):
        """Test that same input produces same heatmap"""
        torch.manual_seed(42)
        input_tensor = torch.randn(1, 1, 224, 224, requires_grad=True)
        
        heatmap1 = explainer_agent.generate_heatmap(input_tensor, target_class=0)
        heatmap2 = explainer_agent.generate_heatmap(input_tensor, target_class=0)
        
        assert np.allclose(heatmap1, heatmap2, atol=1e-6)
        
    def test_multiple_calls(self, explainer_agent):
        """Test multiple consecutive heatmap generations"""
        input_tensor = torch.randn(1, 1, 224, 224, requires_grad=True)
        
        for i in range(5):
            heatmap = explainer_agent.generate_heatmap(input_tensor, target_class=i % 4)
            assert isinstance(heatmap, np.ndarray)
            assert heatmap.shape[0] > 0


class TestIntegrationWithVisionAgent:
    """Test suite for integration with Vision Agent"""
    
    @pytest.fixture
    def vision_model(self):
        """Create vision model"""
        return AlzheimerVisionAgent(num_classes=4)
    
    @pytest.fixture
    def explainer_agent(self, vision_model):
        """Create explainer agent"""
        return RadiomicsExplainerAgent(vision_model)
    
    def test_prediction_and_explanation(self, vision_model, explainer_agent):
        """Test getting prediction and explanation together"""
        input_tensor = torch.randn(1, 1, 224, 224, requires_grad=True)
        
        # Get prediction
        with torch.no_grad():
            output = vision_model(input_tensor)
            predicted_class = output.argmax(dim=1).item()
        
        # Get explanation for predicted class
        heatmap = explainer_agent.generate_heatmap(input_tensor, target_class=predicted_class)
        
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape[0] > 0
        
    def test_explain_all_classes(self, explainer_agent):
        """Test generating explanations for all classes"""
        input_tensor = torch.randn(1, 1, 224, 224, requires_grad=True)
        
        heatmaps = []
        for class_idx in range(4):
            heatmap = explainer_agent.generate_heatmap(input_tensor, target_class=class_idx)
            heatmaps.append(heatmap)
        
        assert len(heatmaps) == 4
        assert all(isinstance(h, np.ndarray) for h in heatmaps)


class TestMemoryManagement:
    """Test suite for memory management"""
    
    @pytest.fixture
    def explainer_agent(self):
        """Create explainer agent"""
        model = AlzheimerVisionAgent(num_classes=4)
        return RadiomicsExplainerAgent(model)
    
    def test_no_memory_leak(self, explainer_agent):
        """Test that repeated calls don't cause memory leaks"""
        input_tensor = torch.randn(1, 1, 224, 224, requires_grad=True)
        
        # Generate many heatmaps
        for i in range(20):
            heatmap = explainer_agent.generate_heatmap(input_tensor, target_class=i % 4)
            del heatmap  # Explicitly delete to help garbage collection
        
        # If we get here without OOM, test passes
        assert True
        
    def test_gradient_cleanup(self, explainer_agent):
        """Test that gradients are properly managed"""
        input_tensor = torch.randn(1, 1, 224, 224, requires_grad=True)
        
        # Generate heatmap
        heatmap = explainer_agent.generate_heatmap(input_tensor, target_class=0)
        
        # Gradients should be detached in output
        assert not isinstance(heatmap, torch.Tensor)
        assert isinstance(heatmap, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
