import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional


class RadiomicsExplainerAgent:
    """
    Agent 4: Radiomics Explainer (Explainable AI / XAI).
    Implements a pure-PyTorch Grad-CAM (Gradient-weighted Class Activation Mapping)
    engine tailored for 2D Grayscale MRI feature visualization.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self.gradients: torch.Tensor = torch.empty(0)
        self.activations: torch.Tensor = torch.empty(0)
        self._register_hooks()

    # Using 'Any' bypasses the broken PyTorch type stubs while maintaining runtime execution
    def _forward_hook(self, module: nn.Module, input_t: Any, output_t: torch.Tensor) -> None:
        self.activations = output_t

    def _backward_hook(self, module: nn.Module, grad_input: Any, grad_output: Any) -> None:
        self.gradients = grad_output[0]

    def _register_hooks(self) -> None:
        target_layer: Optional[nn.Module] = dict(self.model.named_modules()).get("backbone.layer4")
        if target_layer is not None:
            target_layer.register_forward_hook(self._forward_hook)
            # Explicitly silencing Pylance here due to PyTorch's mismatched C++ stubs
            target_layer.register_full_backward_hook(self._backward_hook)  # type: ignore
        else:
            raise AttributeError("[!] Critical: Failed to locate backbone.layer4 in Vision Agent.")

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """Computes the spatial activation heatmap for a target clinical diagnosis."""
        self.model.zero_grad()

        # 1. Run forward pass
        output = self.model(input_tensor)

        # 2. Target the specific clinical class score
        score = output[0][target_class]

        # 3. Execute backward pass to capture gradients flowing through layer4
        score.backward()

        # 4. Extract weights using Global Average Pooling on gradients
        gradients = self.gradients
        activations = self.activations

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # 5. Weight the spatial activation channels
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        # 6. Sum across all channels to get the 2D spatial heatmap
        heatmap_tensor = torch.mean(activations, dim=1).squeeze()

        # 7. Apply ReLU (we only care about features that POSITIVELY correlate with the class)
        heatmap = torch.clamp(heatmap_tensor, min=0).detach().cpu().numpy()

        # 8. Min-Max Intensity Normalization to scale mapping between [0.0, 1.0]
        denom = np.max(heatmap) - np.min(heatmap)
        if denom == 0:
            denom = 1e-8
        heatmap = (heatmap - np.min(heatmap)) / denom

        return heatmap


if __name__ == "__main__":
    from vision_agent import AlzheimerVisionAgent

    print("[*] Testing Radiomics Explainer Hook Registration...")
    mock_model = AlzheimerVisionAgent(num_classes=4)
    explainer = RadiomicsExplainerAgent(mock_model)

    mock_mri_batch = torch.randn(1, 1, 224, 224, requires_grad=True)
    mock_heatmap = explainer.generate_heatmap(mock_mri_batch, target_class=2)

    print(f"[SUCCESS] Grad-CAM Activation Heatmap Generated. Resolution: {mock_heatmap.shape}")
