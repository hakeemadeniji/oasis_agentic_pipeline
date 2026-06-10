"""
Vision Agent Training Script
Comprehensive training pipeline for ResNet18-based Alzheimer's classification model.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import argparse

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(src_dir)

from agents.vision.vision_agent import AlzheimerVisionAgent


class TrainingConfig:
    """Training configuration with hyperparameters"""
    
    def __init__(self, **kwargs):
        # Data paths
        self.data_root = kwargs.get('data_root', 'data/oasis_raw')
        self.output_dir = kwargs.get('output_dir', 'models/vision_agent')
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 'models/checkpoints')
        
        # Model architecture
        self.num_classes = kwargs.get('num_classes', 4)
        self.pretrained = kwargs.get('pretrained', True)
        
        # Training hyperparameters
        self.batch_size = kwargs.get('batch_size', 32)
        self.num_epochs = kwargs.get('num_epochs', 50)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.momentum = kwargs.get('momentum', 0.9)
        
        # Learning rate schedule
        self.lr_scheduler = kwargs.get('lr_scheduler', 'step')
        self.lr_step_size = kwargs.get('lr_step_size', 10)
        self.lr_gamma = kwargs.get('lr_gamma', 0.1)
        
        # Data split
        self.train_split = kwargs.get('train_split', 0.7)
        self.val_split = kwargs.get('val_split', 0.15)
        self.test_split = kwargs.get('test_split', 0.15)
        
        # Data augmentation
        self.use_augmentation = kwargs.get('use_augmentation', True)
        self.rotation_degrees = kwargs.get('rotation_degrees', 15)
        self.horizontal_flip = kwargs.get('horizontal_flip', True)
        
        # Training settings
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        self.save_best_only = kwargs.get('save_best_only', True)
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Logging
        self.log_interval = kwargs.get('log_interval', 10)
        self.tensorboard = kwargs.get('tensorboard', True)
        
        # Random seed
        self.seed = kwargs.get('seed', 42)
        
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class VisionAgentTrainer:
    """Trainer class for Vision Agent model"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds for reproducibility
        self._set_seed(config.seed)
        
        # Create output directories
        self._create_directories()
        
        # Initialize model
        self.model = AlzheimerVisionAgent(num_classes=config.num_classes).to(self.device)
        
        # Initialize data loaders
        self.train_loader, self.val_loader, self.test_loader = self._create_data_loaders()
        
        # Initialize optimizer and loss
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize tensorboard writer
        self.writer = None
        if config.tensorboard:
            log_dir = os.path.join(config.output_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
            self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def _create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders"""
        # Define transforms
        if self.config.use_augmentation:
            train_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((224, 224)),
                transforms.RandomRotation(self.config.rotation_degrees),
                transforms.RandomHorizontalFlip() if self.config.horizontal_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        
        val_test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Load full dataset
        full_dataset = datasets.ImageFolder(root=self.config.data_root)
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(self.config.train_split * total_size)
        val_size = int(self.config.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config.seed)
        )
        
        # Apply transforms
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_test_transform
        test_dataset.dataset.transform = val_test_transform
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        print(f"Class names: {full_dataset.classes}")
        
        return train_loader, val_loader, test_loader
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.lr_scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma
            )
        elif self.config.lr_scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.lr_scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.lr_gamma,
                patience=5
            )
        else:
            return None
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.config.num_epochs}')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{running_loss / (batch_idx + 1):.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def test(self) -> Dict:
        """Test the model and return detailed metrics"""
        self.model.eval()
        correct = 0
        total = 0
        class_correct = [0] * self.config.num_classes
        class_total = [0] * self.config.num_classes
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Per-class accuracy
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += (predicted[i] == labels[i]).item()
                    class_total[label] += 1
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        test_acc = 100. * correct / total
        class_accuracies = [100. * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                           for i in range(self.config.num_classes)]
        
        return {
            'test_accuracy': test_acc,
            'class_accuracies': class_accuracies,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config.to_dict()
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.output_dir, 'best_vision_agent.pth')
            torch.save(self.model.state_dict(), best_path)
            print(f"✓ Saved best model with validation accuracy: {self.best_val_acc:.2f}%")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("Starting Vision Agent Training")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Model: ResNet18 with {self.config.num_classes} classes")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print("="*70 + "\n")
        
        # Save configuration
        config_path = os.path.join(self.config.output_dir, 'training_config.json')
        self.config.save(config_path)
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Check for improvement
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
                break
        
        # Final test evaluation
        print("\n" + "="*70)
        print("Training Complete - Running Final Test Evaluation")
        print("="*70)
        
        # Load best model
        best_model_path = os.path.join(self.config.output_dir, 'best_vision_agent.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        
        # Test
        test_results = self.test()
        
        print(f"\nTest Accuracy: {test_results['test_accuracy']:.2f}%")
        print("\nPer-Class Accuracies:")
        class_names = ["Non Demented", "Very Mild Dementia", "Mild Dementia", "Moderate Dementia"]
        for i, (name, acc) in enumerate(zip(class_names, test_results['class_accuracies'])):
            print(f"  {name}: {acc:.2f}%")
        
        # Save test results
        results_path = os.path.join(self.config.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'test_accuracy': test_results['test_accuracy'],
                'class_accuracies': test_results['class_accuracies'],
                'class_names': class_names
            }, f, indent=4)
        
        # Save training history
        history_path = os.path.join(self.config.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=4)
        
        if self.writer:
            self.writer.close()
        
        print("\n" + "="*70)
        print("Training Pipeline Complete!")
        print(f"Best model saved to: {best_model_path}")
        print(f"Training history saved to: {history_path}")
        print(f"Test results saved to: {results_path}")
        print("="*70 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train Vision Agent for Alzheimer\'s Classification')
    
    # Data arguments
    parser.add_argument('--data-root', type=str, default='data/oasis_raw',
                       help='Path to OASIS dataset root directory')
    parser.add_argument('--output-dir', type=str, default='models/vision_agent',
                       help='Directory to save trained models')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization)')
    
    # Model arguments
    parser.add_argument('--num-classes', type=int, default=4,
                       help='Number of output classes')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_classes=args.num_classes,
        seed=args.seed,
        device=args.device
    )
    
    # Create trainer
    trainer = VisionAgentTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
