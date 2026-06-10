import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from vision_agent import AlzheimerVisionAgent, build_lazy_dataloaders

class VisionAgentTrainer:
    """
    Expert-grade PyTorch Training Pipeline.
    Implements Train/Validation splitting, AdamW optimization, and model checkpointing.
    """
    def __init__(self, data_dir: str, epochs: int = 10, batch_size: int = 32, learning_rate: float = 1e-4):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Hardware optimization: Use GPU/DirectML if available, fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing Training Compute on: {self.device}")

        # 1. Load the full dataset structure
        print("[*] Building Lazy Dataloaders...")
        _, self.full_dataset = build_lazy_dataloaders(data_dir, batch_size)
        self.class_names = self.full_dataset.classes
        
        # 2. Mathematical Splitting: 80% Training, 20% Validation
        total_size = len(self.full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(self.full_dataset, [train_size, val_size])
        
        # 3. Create independent loaders for memory efficiency
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        print(f"[+] Dataset Split -> Training: {train_size} | Validation: {val_size}")

        # 4. Boot the Neural Network Architecture
        self.model = AlzheimerVisionAgent(num_classes=len(self.class_names)).to(self.device)
        
        # 5. Define the mathematical penalization and optimization logic
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3)

    def train_model(self, save_dir: str):
        print(f"\n[*] Commencing Training Protocol: {self.epochs} Epochs")
        best_val_loss = float('inf')
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for epoch in range(self.epochs):
            start_time = time.time()
            
            # --- TRAINING PHASE ---
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()           # Clear old mathematical gradients
                outputs = self.model(images)         # Forward pass (Predict)
                loss = self.criterion(outputs, labels) # Calculate error
                loss.backward()                      # Backpropagation
                self.optimizer.step()                # Update weights
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                if batch_idx % 50 == 0:
                    print(f"    Epoch [{epoch+1}/{self.epochs}] Batch [{batch_idx}/{len(self.train_loader)}] Loss: {loss.item():.4f}")

            train_accuracy = 100 * correct_train / total_train
            train_loss = running_loss / len(self.train_loader)

            # --- VALIDATION PHASE ---
            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            # Disable gradient calculation for validation (saves massive memory)
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_accuracy = 100 * correct_val / total_val
            val_loss = val_loss / len(self.val_loader)
            epoch_time = time.time() - start_time
            
            print(f"\n[Epoch {epoch+1} Summary] Time: {epoch_time:.1f}s")
            print(f" -> Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
            print(f" -> Val Loss:   {val_loss:.4f} | Val Acc:   {val_accuracy:.2f}%")

            # --- MODEL CHECKPOINTING ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(save_dir, "best_vision_agent.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f" [*] New Best Model Checkpoint Saved: {save_path}\n")
            else:
                print("\n")

if __name__ == "__main__":
    # Paths are now correctly anchored to the root folder of your project
    RAW_DATA_DIR = os.path.join("data", "oasis_raw")
    MODEL_SAVE_DIR = os.path.join("src", "pipeline", "onnx_inference")
    
    trainer = VisionAgentTrainer(data_dir=RAW_DATA_DIR, epochs=5, batch_size=32)
    trainer.train_model(save_dir=MODEL_SAVE_DIR)
    