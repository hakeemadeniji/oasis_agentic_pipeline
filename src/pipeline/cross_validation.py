"""
Cross-Validation Pipeline for Vision Agent
Implements K-Fold cross-validation for robust model evaluation.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
from typing import Dict, List, Tuple
import argparse
from sklearn.model_selection import KFold, StratifiedKFold

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(src_dir)

from agents.vision.vision_agent import AlzheimerVisionAgent
from pipeline.train_vision_agent import TrainingConfig


class CrossValidationPipeline:
    """K-Fold Cross-Validation for Vision Agent"""
    
    def __init__(
        self,
        config: TrainingConfig,
        n_folds: int = 5,
        stratified: bool = True,
        shuffle: bool = True
    ):
        self.config = config
        self.n_folds = n_folds
        self.stratified = stratified
        self.shuffle = shuffle
        self.device = torch.device(config.device)
        
        # Set random seed
        self._set_seed(config.seed)
        
        # Create output directory
        self.cv_output_dir = os.path.join(config.output_dir, 'cross_validation')
        os.makedirs(self.cv_output_dir, exist_ok=True)
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Create fold splitter
        self.kfold = self._create_kfold()
        
        # Results storage
        self.fold_results = []
        
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def _load_dataset(self):
        """Load full dataset"""
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        dataset = datasets.ImageFolder(root=self.config.data_root, transform=transform)
        print(f"Loaded dataset with {len(dataset)} samples")
        print(f"Classes: {dataset.classes}")
        
        return dataset
    
    def _create_kfold(self):
        """Create K-Fold splitter"""
        if self.stratified:
            # Get labels for stratification
            labels = [self.dataset.targets[i] for i in range(len(self.dataset))]
            return StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.config.seed
            )
        else:
            return KFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.config.seed
            )
    
    def _create_data_loaders(
        self,
        train_indices: List[int],
        val_indices: List[int]
    ) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders for a fold"""
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_fold(
        self,
        fold: int,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict:
        """Train model for one fold"""
        print(f"\n{'='*70}")
        print(f"Training Fold {fold + 1}/{self.n_folds}")
        print(f"{'='*70}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        
        # Initialize model
        model = AlzheimerVisionAgent(num_classes=self.config.num_classes).to(self.device)
        
        # Initialize optimizer and loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        # Initialize scheduler
        if self.config.lr_scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma
            )
        elif self.config.lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.num_epochs
            )
        else:
            scheduler = None
        
        # Training state
        best_val_acc = 0.0
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Fold {fold + 1} Epoch {epoch + 1}')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{train_loss / (pbar.n + 1):.4f}',
                    'acc': f'{100. * train_correct / train_total:.2f}%'
                })
            
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = 100. * train_correct / train_total
            
            # Validate
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = 100. * val_correct / val_total
            
            # Update history
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc)
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Check for improvement
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_val_loss = epoch_val_loss
                epochs_without_improvement = 0
                
                # Save best model for this fold
                fold_model_path = os.path.join(self.cv_output_dir, f'fold_{fold + 1}_best.pth')
                torch.save(model.state_dict(), fold_model_path)
            else:
                epochs_without_improvement += 1
            
            # Print epoch summary
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}: Train Acc: {epoch_train_acc:.2f}%, Val Acc: {epoch_val_acc:.2f}%")
            
            # Early stopping
            if epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        return {
            'fold': fold + 1,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'history': history
        }
    
    def run_cross_validation(self) -> Dict:
        """Run complete cross-validation"""
        print("\n" + "="*70)
        print("Starting Cross-Validation")
        print("="*70)
        print(f"Number of folds: {self.n_folds}")
        print(f"Stratified: {self.stratified}")
        print(f"Total samples: {len(self.dataset)}")
        print("="*70 + "\n")
        
        # Get labels for stratification
        if self.stratified:
            labels = np.array([self.dataset.targets[i] for i in range(len(self.dataset))])
            splits = self.kfold.split(np.zeros(len(self.dataset)), labels)
        else:
            splits = self.kfold.split(np.zeros(len(self.dataset)))
        
        # Run each fold
        for fold, (train_indices, val_indices) in enumerate(splits):
            # Create data loaders
            train_loader, val_loader = self._create_data_loaders(train_indices, val_indices)
            
            # Train fold
            fold_result = self.train_fold(fold, train_loader, val_loader)
            self.fold_results.append(fold_result)
            
            # Save fold results
            fold_results_path = os.path.join(self.cv_output_dir, f'fold_{fold + 1}_results.json')
            with open(fold_results_path, 'w') as f:
                json.dump(fold_result, f, indent=4)
        
        # Calculate aggregate statistics
        aggregate_results = self._calculate_aggregate_statistics()
        
        # Save aggregate results
        self._save_aggregate_results(aggregate_results)
        
        return aggregate_results
    
    def _calculate_aggregate_statistics(self) -> Dict:
        """Calculate aggregate statistics across folds"""
        val_accs = [result['best_val_acc'] for result in self.fold_results]
        val_losses = [result['best_val_loss'] for result in self.fold_results]
        
        return {
            'n_folds': self.n_folds,
            'mean_val_acc': np.mean(val_accs),
            'std_val_acc': np.std(val_accs),
            'min_val_acc': np.min(val_accs),
            'max_val_acc': np.max(val_accs),
            'mean_val_loss': np.mean(val_losses),
            'std_val_loss': np.std(val_losses),
            'fold_results': self.fold_results
        }
    
    def _save_aggregate_results(self, aggregate_results: Dict):
        """Save aggregate results"""
        print("\n" + "="*70)
        print("Cross-Validation Complete")
        print("="*70)
        print(f"Mean Validation Accuracy: {aggregate_results['mean_val_acc']:.2f}% ± {aggregate_results['std_val_acc']:.2f}%")
        print(f"Min Validation Accuracy: {aggregate_results['min_val_acc']:.2f}%")
        print(f"Max Validation Accuracy: {aggregate_results['max_val_acc']:.2f}%")
        print(f"Mean Validation Loss: {aggregate_results['mean_val_loss']:.4f} ± {aggregate_results['std_val_loss']:.4f}")
        
        print("\nPer-Fold Results:")
        for result in self.fold_results:
            print(f"  Fold {result['fold']}: {result['best_val_acc']:.2f}%")
        
        # Save to file
        results_path = os.path.join(self.cv_output_dir, 'cross_validation_results.json')
        with open(results_path, 'w') as f:
            json.dump(aggregate_results, f, indent=4)
        
        print(f"\n✓ Results saved to: {results_path}")
        print("="*70 + "\n")


class LeaveOneOutCV:
    """Leave-One-Out Cross-Validation (for small datasets)"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print("Leave-One-Out CV is computationally expensive.")
        print("Only recommended for very small datasets (< 100 samples).")
        print("Consider using K-Fold CV instead.")


class TimeSeriesCV:
    """Time-Series Cross-Validation for longitudinal data"""
    
    def __init__(self, config: TrainingConfig, n_splits: int = 5):
        self.config = config
        self.n_splits = n_splits
        self.device = torch.device(config.device)
        
        print("Time-Series CV maintains temporal order in splits.")
        print("Useful for longitudinal patient data.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Cross-Validation for Vision Agent')
    
    parser.add_argument('--data-root', type=str, default='data/oasis_raw',
                       help='Path to OASIS dataset root directory')
    parser.add_argument('--output-dir', type=str, default='models/cross_validation',
                       help='Directory to save CV results')
    parser.add_argument('--n-folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--stratified', action='store_true', default=True,
                       help='Use stratified K-Fold')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs per fold')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        num_classes=4,
        seed=args.seed,
        device=args.device,
        tensorboard=False  # Disable for CV
    )
    
    # Run cross-validation
    cv_pipeline = CrossValidationPipeline(
        config=config,
        n_folds=args.n_folds,
        stratified=args.stratified
    )
    
    results = cv_pipeline.run_cross_validation()


if __name__ == '__main__':
    main()
