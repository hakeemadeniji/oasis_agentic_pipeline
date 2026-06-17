import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from typing import Tuple


class AlzheimerVisionAgent(nn.Module):
    """
    Advanced PyTorch Vision Agent.
    Utilizes a ResNet backbone customized for clinical MRI slices.
    """

    def __init__(self, num_classes=4):
        super().__init__()

        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

    def forward(self, x_tensor):
        return self.backbone(x_tensor)


# STRICT TYPE HINTS ADDED HERE TO APPEASE PYLANCE
def build_lazy_dataloaders(
    data_dir: str, batch_size: int = 32
) -> Tuple[DataLoader, datasets.ImageFolder]:
    """
    Constructs the Lazy-Loading Pipeline.
    Images remain on the SSD until they are mathematically transformed in RAM.
    """
    transform_pipeline = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root=data_dir, transform=transform_pipeline)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return loader, dataset


if __name__ == "__main__":
    DATA_DIR = os.path.join("data", "oasis_raw")

    print("[*] Initializing Lazy Loading Engine...")
    loader, dataset = build_lazy_dataloaders(DATA_DIR, batch_size=16)

    # Class names are now safely extracted from the dataset object
    class_names = dataset.classes

    print(f"[+] Dynamically mapped classes: {class_names}")
    print(f"[+] Total batches prepared per epoch: {len(loader)}")

    print("\n[*] Booting Vision Agent Architecture...")
    agent = AlzheimerVisionAgent(num_classes=len(class_names))

    dummy_batch = torch.randn(16, 1, 224, 224)
    output_predictions = agent(dummy_batch)

    print(f"[SUCCESS] Vision Agent Output Tensor Shape: {output_predictions.shape}")
