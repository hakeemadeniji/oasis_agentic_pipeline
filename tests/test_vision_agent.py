import pytest
import torch
import numpy as np
from PIL import Image
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.vision.vision_agent import AlzheimerVisionAgent, build_lazy_dataloaders


class TestAlzheimerVisionAgent:
    """Test suite for Vision Agent (Agent 1)"""
    
    @pytest.fixture
    def vision_agent(self):
        """Create a vision agent instance for testing"""
        return AlzheimerVisionAgent(num_classes=4)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor"""
        return torch.randn(1, 1, 224, 224)
    
    def test_agent_initialization(self, vision_agent):
        """Test that agent initializes correctly"""
        assert vision_agent is not None
        assert isinstance(vision_agent, torch.nn.Module)
        
    def test_forward_pass_shape(self, vision_agent, sample_input):
        """Test forward pass returns correct output shape"""
        output = vision_agent(sample_input)
        assert output.shape == (1, 4), f"Expected shape (1, 4), got {output.shape}"
        
    def test_forward_pass_no_nan(self, vision_agent, sample_input):
        """Test forward pass doesn't produce NaN values"""
        output = vision_agent(sample_input)
        assert not torch.isnan(output).any(), "Output contains NaN values"
        
    def test_forward_pass_no_inf(self, vision_agent, sample_input):
        """Test forward pass doesn't produce infinite values"""
        output = vision_agent(sample_input)
        assert not torch.isinf(output).any(), "Output contains infinite values"
        
    def test_batch_processing(self, vision_agent):
        """Test agent can process batches"""
        batch_input = torch.randn(8, 1, 224, 224)
        output = vision_agent(batch_input)
        assert output.shape == (8, 4), f"Expected shape (8, 4), got {output.shape}"
        
    def test_gradient_flow(self, vision_agent, sample_input):
        """Test gradients flow through the network"""
        sample_input.requires_grad = True
        output = vision_agent(sample_input)
        loss = output.sum()
        loss.backward()
        assert sample_input.grad is not None, "Gradients not computed"
        
    def test_eval_mode(self, vision_agent, sample_input):
        """Test agent works in eval mode"""
        vision_agent.eval()
        with torch.no_grad():
            output = vision_agent(sample_input)
        assert output.shape == (1, 4)
        
    def test_device_transfer(self, vision_agent, sample_input):
        """Test agent can be moved to different devices"""
        device = torch.device("cpu")
        vision_agent = vision_agent.to(device)
        sample_input = sample_input.to(device)
        output = vision_agent(sample_input)
        assert output.device == device
        
    def test_softmax_probabilities(self, vision_agent, sample_input):
        """Test softmax probabilities sum to 1"""
        output = vision_agent(sample_input)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
        
    def test_prediction_consistency(self, vision_agent, sample_input):
        """Test predictions are consistent in eval mode"""
        vision_agent.eval()
        with torch.no_grad():
            output1 = vision_agent(sample_input)
            output2 = vision_agent(sample_input)
        assert torch.allclose(output1, output2), "Predictions not consistent"


class TestDataLoaders:
    """Test suite for data loading functionality"""
    
    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary data directory structure"""
        # Create class directories
        classes = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
        for cls in classes:
            cls_dir = tmp_path / cls
            cls_dir.mkdir()
            
            # Create dummy images
            for i in range(3):
                img = Image.new('L', (224, 224), color=128)
                img.save(cls_dir / f"image_{i}.jpg")
        
        return str(tmp_path)
    
    def test_dataloader_creation(self, temp_data_dir):
        """Test dataloader can be created"""
        loader, dataset = build_lazy_dataloaders(temp_data_dir, batch_size=2)
        assert loader is not None
        assert dataset is not None
        
    def test_dataset_classes(self, temp_data_dir):
        """Test dataset correctly identifies classes"""
        loader, dataset = build_lazy_dataloaders(temp_data_dir, batch_size=2)
        expected_classes = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]
        assert sorted(dataset.classes) == sorted(expected_classes)
        
    def test_dataloader_iteration(self, temp_data_dir):
        """Test dataloader can be iterated"""
        loader, dataset = build_lazy_dataloaders(temp_data_dir, batch_size=2)
        batch_count = 0
        for images, labels in loader:
            batch_count += 1
            assert images.shape[1:] == (1, 224, 224), f"Unexpected image shape: {images.shape}"
            assert labels.shape[0] <= 2, f"Batch size exceeded: {labels.shape[0]}"
        assert batch_count > 0, "No batches loaded"
        
    def test_image_transform(self, temp_data_dir):
        """Test images are correctly transformed"""
        loader, dataset = build_lazy_dataloaders(temp_data_dir, batch_size=1)
        images, labels = next(iter(loader))
        
        # Check shape
        assert images.shape == (1, 1, 224, 224)
        
        # Check value range (should be [0, 1] after ToTensor)
        assert images.min() >= 0.0
        assert images.max() <= 1.0


class TestModelPersistence:
    """Test suite for model saving and loading"""
    
    @pytest.fixture
    def vision_agent(self):
        return AlzheimerVisionAgent(num_classes=4)
    
    def test_save_load_state_dict(self, vision_agent, tmp_path):
        """Test model can be saved and loaded"""
        save_path = tmp_path / "test_model.pth"
        
        # Save model
        torch.save(vision_agent.state_dict(), save_path)
        assert save_path.exists()
        
        # Load model
        new_agent = AlzheimerVisionAgent(num_classes=4)
        new_agent.load_state_dict(torch.load(save_path))
        
        # Test predictions match
        sample_input = torch.randn(1, 1, 224, 224)
        with torch.no_grad():
            output1 = vision_agent(sample_input)
            output2 = new_agent(sample_input)
        
        assert torch.allclose(output1, output2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
