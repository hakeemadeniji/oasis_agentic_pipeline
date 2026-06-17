"""
Model Versioning and Checkpointing System
Comprehensive system for tracking, managing, and versioning trained models.
"""

import os
import sys
import torch
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(src_dir)


class ModelVersion:
    """Represents a single model version"""
    
    def __init__(
        self,
        version_id: str,
        model_name: str,
        model_path: str,
        metadata: Dict[str, Any]
    ):
        self.version_id = version_id
        self.model_name = model_name
        self.model_path = model_path
        self.metadata = metadata
        self.created_at = metadata.get('created_at', datetime.now().isoformat())
        
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'version_id': self.version_id,
            'model_name': self.model_name,
            'model_path': self.model_path,
            'metadata': self.metadata,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create from dictionary"""
        return cls(
            version_id=data['version_id'],
            model_name=data['model_name'],
            model_path=data['model_path'],
            metadata=data['metadata']
        )


class ModelRegistry:
    """Central registry for managing model versions"""
    
    def __init__(self, registry_dir: str = 'models/registry'):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.registry_dir / 'model_registry.json'
        self.versions: Dict[str, List[ModelVersion]] = {}
        
        # Load existing registry
        self._load_registry()
        
    def _load_registry(self):
        """Load registry from disk"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                for model_name, versions in data.items():
                    self.versions[model_name] = [
                        ModelVersion.from_dict(v) for v in versions
                    ]
        
    def _save_registry(self):
        """Save registry to disk"""
        data = {}
        for model_name, versions in self.versions.items():
            data[model_name] = [v.to_dict() for v in versions]
        
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=4)
    
    def register_model(
        self,
        model_name: str,
        model_path: str,
        metadata: Dict[str, Any],
        version_id: Optional[str] = None
    ) -> ModelVersion:
        """Register a new model version"""
        # Generate version ID if not provided
        if version_id is None:
            version_id = self._generate_version_id(model_name)
        
        # Add timestamp to metadata
        metadata['created_at'] = datetime.now().isoformat()
        metadata['registered_by'] = 'ModelRegistry'
        
        # Create model version
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            model_path=model_path,
            metadata=metadata
        )
        
        # Add to registry
        if model_name not in self.versions:
            self.versions[model_name] = []
        self.versions[model_name].append(version)
        
        # Save registry
        self._save_registry()
        
        print(f"✓ Registered model: {model_name} (version: {version_id})")
        return version
    
    def _generate_version_id(self, model_name: str) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        existing_versions = len(self.versions.get(model_name, []))
        return f"v{existing_versions + 1}_{timestamp}"
    
    def get_model(self, model_name: str, version_id: Optional[str] = None) -> Optional[ModelVersion]:
        """Get a specific model version"""
        if model_name not in self.versions:
            return None
        
        versions = self.versions[model_name]
        
        if version_id is None:
            # Return latest version
            return versions[-1] if versions else None
        
        # Find specific version
        for version in versions:
            if version.version_id == version_id:
                return version
        
        return None
    
    def list_models(self, model_name: Optional[str] = None) -> List[ModelVersion]:
        """List all models or versions of a specific model"""
        if model_name:
            return self.versions.get(model_name, [])
        
        # Return all versions of all models
        all_versions = []
        for versions in self.versions.values():
            all_versions.extend(versions)
        return all_versions
    
    def delete_version(self, model_name: str, version_id: str) -> bool:
        """Delete a specific model version"""
        if model_name not in self.versions:
            return False
        
        versions = self.versions[model_name]
        for i, version in enumerate(versions):
            if version.version_id == version_id:
                # Remove from registry
                versions.pop(i)
                
                # Delete model file if it exists
                if os.path.exists(version.model_path):
                    os.remove(version.model_path)
                    print(f"✓ Deleted model file: {version.model_path}")
                
                # Save registry
                self._save_registry()
                print(f"✓ Deleted version: {version_id}")
                return True
        
        return False
    
    def promote_to_production(self, model_name: str, version_id: str) -> bool:
        """Promote a model version to production"""
        version = self.get_model(model_name, version_id)
        if not version:
            return False
        
        # Update metadata
        version.metadata['stage'] = 'production'
        version.metadata['promoted_at'] = datetime.now().isoformat()
        
        # Demote other production versions
        for v in self.versions[model_name]:
            if v.version_id != version_id and v.metadata.get('stage') == 'production':
                v.metadata['stage'] = 'archived'
        
        self._save_registry()
        print(f"✓ Promoted {model_name} version {version_id} to production")
        return True
    
    def get_production_model(self, model_name: str) -> Optional[ModelVersion]:
        """Get the current production model"""
        if model_name not in self.versions:
            return None
        
        for version in reversed(self.versions[model_name]):
            if version.metadata.get('stage') == 'production':
                return version
        
        return None
    
    def compare_versions(self, model_name: str, version_id1: str, version_id2: str) -> Dict:
        """Compare two model versions"""
        v1 = self.get_model(model_name, version_id1)
        v2 = self.get_model(model_name, version_id2)
        
        if not v1 or not v2:
            return {}
        
        comparison = {
            'version_1': {
                'id': v1.version_id,
                'created_at': v1.created_at,
                'metadata': v1.metadata
            },
            'version_2': {
                'id': v2.version_id,
                'created_at': v2.created_at,
                'metadata': v2.metadata
            },
            'differences': {}
        }
        
        # Compare metrics
        for key in v1.metadata.keys():
            if key in v2.metadata:
                if v1.metadata[key] != v2.metadata[key]:
                    comparison['differences'][key] = {
                        'v1': v1.metadata[key],
                        'v2': v2.metadata[key]
                    }
        
        return comparison


class CheckpointManager:
    """Manages model checkpoints during training"""
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best_only: bool = False
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        
        self.checkpoints: List[Dict] = []
        self.best_metric = None
        
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None
    ) -> str:
        """Save a checkpoint"""
        # Determine if this is the best checkpoint
        current_metric = metrics.get('val_acc', 0)
        is_best = self.best_metric is None or current_metric > self.best_metric
        
        if is_best:
            self.best_metric = current_metric
        
        # Skip if save_best_only and not best
        if self.save_best_only and not is_best:
            return ""
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate checkpoint filename
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Track checkpoint
        self.checkpoints.append({
            'path': str(checkpoint_path),
            'epoch': epoch,
            'metrics': metrics,
            'is_best': is_best
        })
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(model.state_dict(), best_path)
            print(f"✓ Saved best model (epoch {epoch}, val_acc: {current_metric:.2f}%)")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        return str(checkpoint_path)
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit"""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by epoch
        self.checkpoints.sort(key=lambda x: x['epoch'])
        
        # Remove oldest checkpoints (keep best)
        while len(self.checkpoints) > self.max_checkpoints:
            checkpoint = self.checkpoints[0]
            if not checkpoint['is_best']:
                if os.path.exists(checkpoint['path']):
                    os.remove(checkpoint['path'])
                self.checkpoints.pop(0)
            else:
                # Don't remove best checkpoint
                break
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict:
        """Load a checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    
    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Optional[Dict]:
        """Load the best checkpoint"""
        best_path = self.checkpoint_dir / 'best_model.pth'
        
        if not best_path.exists():
            print("⚠ No best model checkpoint found")
            return None
        
        model.load_state_dict(torch.load(best_path, map_location='cpu'))
        print("✓ Loaded best model checkpoint")
        
        return {'model_state_dict': model.state_dict()}
    
    def list_checkpoints(self) -> List[Dict]:
        """List all checkpoints"""
        return self.checkpoints


class ModelArtifactManager:
    """Manages model artifacts (weights, configs, metadata)"""
    
    def __init__(self, artifacts_dir: str = 'models/artifacts'):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
    def save_artifact(
        self,
        model_name: str,
        version_id: str,
        model_state_dict: Dict,
        config: Dict,
        metrics: Dict,
        additional_files: Optional[Dict[str, str]] = None
    ) -> str:
        """Save complete model artifact"""
        # Create version directory
        version_dir = self.artifacts_dir / model_name / version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_path = version_dir / 'model.pth'
        torch.save(model_state_dict, model_path)
        
        # Save configuration
        config_path = version_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Save metrics
        metrics_path = version_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save additional files
        if additional_files:
            for filename, filepath in additional_files.items():
                dest_path = version_dir / filename
                shutil.copy2(filepath, dest_path)
        
        # Create manifest
        manifest = {
            'model_name': model_name,
            'version_id': version_id,
            'created_at': datetime.now().isoformat(),
            'files': {
                'model': str(model_path),
                'config': str(config_path),
                'metrics': str(metrics_path)
            }
        }
        
        manifest_path = version_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=4)
        
        print(f"✓ Saved artifact: {model_name}/{version_id}")
        return str(version_dir)
    
    def load_artifact(self, model_name: str, version_id: str) -> Dict:
        """Load complete model artifact"""
        version_dir = self.artifacts_dir / model_name / version_id
        
        if not version_dir.exists():
            raise FileNotFoundError(f"Artifact not found: {model_name}/{version_id}")
        
        # Load manifest
        manifest_path = version_dir / 'manifest.json'
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Load model weights
        model_path = version_dir / 'model.pth'
        model_state_dict = torch.load(model_path, map_location='cpu')
        
        # Load config
        config_path = version_dir / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load metrics
        metrics_path = version_dir / 'metrics.json'
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        return {
            'manifest': manifest,
            'model_state_dict': model_state_dict,
            'config': config,
            'metrics': metrics
        }


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(description='Model Versioning and Management')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register a new model version')
    register_parser.add_argument('--model-name', required=True, help='Model name')
    register_parser.add_argument('--model-path', required=True, help='Path to model file')
    register_parser.add_argument('--metadata', type=str, help='Metadata JSON string')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List model versions')
    list_parser.add_argument('--model-name', help='Model name (optional)')
    
    # Promote command
    promote_parser = subparsers.add_parser('promote', help='Promote version to production')
    promote_parser.add_argument('--model-name', required=True, help='Model name')
    promote_parser.add_argument('--version-id', required=True, help='Version ID')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two versions')
    compare_parser.add_argument('--model-name', required=True, help='Model name')
    compare_parser.add_argument('--version-id1', required=True, help='First version ID')
    compare_parser.add_argument('--version-id2', required=True, help='Second version ID')
    
    args = parser.parse_args()
    
    # Initialize registry
    registry = ModelRegistry()
    
    if args.command == 'register':
        metadata = json.loads(args.metadata) if args.metadata else {}
        registry.register_model(args.model_name, args.model_path, metadata)
        
    elif args.command == 'list':
        versions = registry.list_models(args.model_name)
        print(f"\nFound {len(versions)} version(s):")
        for v in versions:
            print(f"  {v.model_name} - {v.version_id} (created: {v.created_at})")
            
    elif args.command == 'promote':
        registry.promote_to_production(args.model_name, args.version_id)
        
    elif args.command == 'compare':
        comparison = registry.compare_versions(
            args.model_name,
            args.version_id1,
            args.version_id2
        )
        print(json.dumps(comparison, indent=4))


if __name__ == '__main__':
    main()
