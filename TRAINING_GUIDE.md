# OASIS Agentic Pipeline - Model Training Guide

**Version:** 1.0  
**Last Updated:** June 10, 2026  
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Training Pipeline](#training-pipeline)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Cross-Validation](#cross-validation)
7. [Model Versioning](#model-versioning)
8. [ONNX Export](#onnx-export)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)
11. [Performance Optimization](#performance-optimization)

---

## Overview

This guide provides comprehensive instructions for training, optimizing, and deploying the Vision Agent model in the OASIS Agentic Pipeline. The training infrastructure includes:

- **Automated Training Pipeline**: Complete end-to-end training with data loading, augmentation, and validation
- **Hyperparameter Tuning**: Bayesian optimization using Optuna for optimal model configuration
- **Cross-Validation**: K-Fold and Stratified K-Fold for robust model evaluation
- **Model Versioning**: Comprehensive tracking and management of model versions
- **ONNX Export**: Production-ready model export for deployment

---

## Prerequisites

### System Requirements

**Minimum:**
- Python 3.14+
- 8GB RAM
- 10GB free disk space
- CPU with 4+ cores

**Recommended:**
- Python 3.14+
- 16GB+ RAM
- 50GB+ free disk space (SSD preferred)
- NVIDIA GPU with 6GB+ VRAM
- CUDA 11.8+ (for GPU training)

### Software Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - Computer vision utilities
- `optuna>=3.0.0` - Hyperparameter optimization
- `onnx>=1.14.0` - ONNX model format
- `onnxruntime>=1.15.0` - ONNX inference engine
- `scikit-learn>=1.3.0` - Machine learning utilities
- `tensorboard>=2.13.0` - Training visualization

### Data Preparation

Ensure your OASIS dataset is organized as follows:

```
data/oasis_raw/
├── Non Demented/           # Class 0: Healthy controls
├── Very mild Dementia/     # Class 1: CDR 0.5
├── Mild Dementia/          # Class 2: CDR 1
└── Moderate Dementia/      # Class 3: CDR 2
```

Each directory should contain MRI brain scan images in standard formats (PNG, JPG, etc.).

---

## Quick Start

### Basic Training

Train a model with default settings:

```bash
python src/pipeline/train_vision_agent.py \
    --data-root data/oasis_raw \
    --output-dir models/vision_agent \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001
```

### Training with GPU

```bash
python src/pipeline/train_vision_agent.py \
    --data-root data/oasis_raw \
    --output-dir models/vision_agent \
    --device cuda \
    --epochs 50 \
    --batch-size 64
```

### Resume Training from Checkpoint

```bash
python src/pipeline/train_vision_agent.py \
    --data-root data/oasis_raw \
    --output-dir models/vision_agent \
    --resume models/checkpoints/latest_checkpoint.pth
```

---

## Training Pipeline

### Architecture

The Vision Agent uses a **ResNet18** architecture modified for single-channel (grayscale) MRI input:

- **Input**: 1-channel grayscale images (224x224 pixels)
- **Backbone**: ResNet18 with modified first convolutional layer
- **Output**: 4-class softmax (Non Demented, Very Mild, Mild, Moderate)
- **Parameters**: ~11M trainable parameters

### Training Configuration

#### Default Hyperparameters

```python
config = TrainingConfig(
    # Data
    data_root='data/oasis_raw',
    batch_size=32,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    
    # Model
    num_classes=4,
    pretrained=True,
    
    # Optimization
    learning_rate=0.001,
    weight_decay=1e-4,
    momentum=0.9,
    num_epochs=50,
    
    # Learning rate schedule
    lr_scheduler='step',
    lr_step_size=10,
    lr_gamma=0.1,
    
    # Data augmentation
    use_augmentation=True,
    rotation_degrees=15,
    horizontal_flip=True,
    
    # Training
    early_stopping_patience=10,
    device='cuda'
)
```

#### Custom Configuration

Create a custom configuration file:

```python
from pipeline.train_vision_agent import TrainingConfig

config = TrainingConfig(
    data_root='data/oasis_raw',
    output_dir='models/custom_training',
    batch_size=64,
    learning_rate=0.0005,
    num_epochs=100,
    lr_scheduler='cosine',
    use_augmentation=True,
    rotation_degrees=20
)

# Save configuration
config.save('models/custom_training/config.json')
```

### Data Augmentation

The training pipeline supports various augmentation techniques:

**Enabled by default:**
- Random rotation (±15 degrees)
- Random horizontal flip
- Grayscale normalization (mean=0.5, std=0.5)

**Disable augmentation:**
```bash
python src/pipeline/train_vision_agent.py \
    --data-root data/oasis_raw \
    --use-augmentation False
```

### Learning Rate Schedules

Three learning rate schedulers are available:

1. **Step Decay** (default):
   - Reduces LR by factor of 0.1 every 10 epochs
   - Best for stable convergence

2. **Cosine Annealing**:
   - Smooth LR decay following cosine curve
   - Good for fine-tuning

3. **Reduce on Plateau**:
   - Reduces LR when validation loss plateaus
   - Adaptive to training dynamics

Example:
```bash
python src/pipeline/train_vision_agent.py \
    --lr-scheduler cosine \
    --epochs 50
```

### Monitoring Training

#### TensorBoard

Training metrics are automatically logged to TensorBoard:

```bash
tensorboard --logdir models/vision_agent/logs
```

View in browser at `http://localhost:6006`

**Logged metrics:**
- Training loss and accuracy
- Validation loss and accuracy
- Learning rate
- Per-class accuracies

#### Training Output

The training script provides real-time progress:

```
Epoch 1/50
Train Loss: 1.2345 | Train Acc: 45.67%
Val Loss: 1.1234 | Val Acc: 52.34%
Learning Rate: 0.001000

✓ Saved best model with validation accuracy: 52.34%
```

### Output Files

After training, the following files are generated:

```
models/vision_agent/
├── best_vision_agent.pth          # Best model weights
├── training_config.json           # Training configuration
├── training_history.json          # Loss/accuracy history
├── test_results.json              # Final test metrics
└── logs/                          # TensorBoard logs
    └── 20260610_120000/
```

---

## Hyperparameter Tuning

### Bayesian Optimization with Optuna

Automatically find optimal hyperparameters:

```bash
python src/pipeline/hyperparameter_tuning.py \
    --data-root data/oasis_raw \
    --output-dir models/tuning \
    --n-trials 50 \
    --method bayesian
```

### Tuned Hyperparameters

The tuning framework optimizes:

- **Batch size**: [16, 32, 64]
- **Learning rate**: [1e-5, 1e-2] (log scale)
- **Weight decay**: [1e-6, 1e-3] (log scale)
- **Momentum**: [0.8, 0.99]
- **LR scheduler**: [step, cosine, plateau]
- **Augmentation settings**: rotation degrees, flip probability

### Tuning Methods

#### 1. Bayesian Optimization (Recommended)

Uses Tree-structured Parzen Estimator (TPE) for efficient search:

```bash
python src/pipeline/hyperparameter_tuning.py \
    --method bayesian \
    --n-trials 50
```

**Advantages:**
- Efficient exploration of hyperparameter space
- Learns from previous trials
- Typically finds good solutions in 30-50 trials

#### 2. Random Search

Random sampling of hyperparameter space:

```bash
python src/pipeline/hyperparameter_tuning.py \
    --method random \
    --n-trials 20
```

**Advantages:**
- Simple and robust
- Good baseline for comparison
- Parallelizable

#### 3. Grid Search

Exhaustive search over predefined grid:

```bash
python src/pipeline/hyperparameter_tuning.py \
    --method grid
```

**Note:** Grid search is computationally expensive and not recommended for large search spaces.

### Pruning

The tuning framework uses **MedianPruner** to stop unpromising trials early:

- Monitors validation accuracy during training
- Prunes trials performing worse than median
- Saves computational resources

### Results Analysis

After tuning, analyze results:

```bash
# View best hyperparameters
cat models/tuning/best_hyperparameters.json

# View all trials
cat models/tuning/all_trials.json

# Open visualization
open models/tuning/optimization_history.html
open models/tuning/param_importances.html
```

### Train with Best Hyperparameters

```bash
python src/pipeline/hyperparameter_tuning.py \
    --data-root data/oasis_raw \
    --output-dir models/tuning \
    --n-trials 50 \
    --train-best
```

This will:
1. Run hyperparameter tuning
2. Automatically train a full model with best parameters
3. Save the final model

---

## Cross-Validation

### K-Fold Cross-Validation

Evaluate model robustness with K-Fold CV:

```bash
python src/pipeline/cross_validation.py \
    --data-root data/oasis_raw \
    --output-dir models/cross_validation \
    --n-folds 5 \
    --epochs 30
```

### Stratified K-Fold (Recommended)

Maintains class distribution in each fold:

```bash
python src/pipeline/cross_validation.py \
    --data-root data/oasis_raw \
    --n-folds 5 \
    --stratified \
    --epochs 30
```

**Why stratified?**
- Ensures balanced class representation
- More reliable performance estimates
- Critical for imbalanced datasets

### Cross-Validation Results

The CV pipeline generates:

```
models/cross_validation/
├── cross_validation_results.json  # Aggregate statistics
├── fold_1_best.pth               # Best model from fold 1
├── fold_1_results.json           # Fold 1 metrics
├── fold_2_best.pth
├── fold_2_results.json
└── ...
```

**Aggregate statistics:**
```json
{
    "n_folds": 5,
    "mean_val_acc": 87.45,
    "std_val_acc": 2.31,
    "min_val_acc": 84.23,
    "max_val_acc": 90.12
}
```

### Interpreting CV Results

**Good model:**
- Mean accuracy > 85%
- Low standard deviation (< 3%)
- Consistent performance across folds

**Overfitting indicators:**
- High variance between folds
- Large gap between train and validation accuracy

**Underfitting indicators:**
- Low accuracy across all folds
- Similar train and validation accuracy (both low)

---

## Model Versioning

### Model Registry

Track and manage model versions:

```bash
# Register a new model
python src/pipeline/model_versioning.py register \
    --model-name vision_agent \
    --model-path models/vision_agent/best_vision_agent.pth \
    --metadata '{"accuracy": 87.5, "epoch": 45}'

# List all versions
python src/pipeline/model_versioning.py list \
    --model-name vision_agent

# Promote to production
python src/pipeline/model_versioning.py promote \
    --model-name vision_agent \
    --version-id v1_20260610_120000
```

### Checkpoint Management

Automatic checkpoint saving during training:

```python
from pipeline.model_versioning import CheckpointManager

checkpoint_manager = CheckpointManager(
    checkpoint_dir='models/checkpoints',
    max_checkpoints=5,
    save_best_only=False
)

# During training
checkpoint_manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    metrics={'val_acc': 87.5, 'val_loss': 0.345}
)
```

**Features:**
- Automatic cleanup of old checkpoints
- Preserves best model
- Stores optimizer state for resuming

### Model Artifacts

Save complete model artifacts:

```python
from pipeline.model_versioning import ModelArtifactManager

artifact_manager = ModelArtifactManager('models/artifacts')

artifact_manager.save_artifact(
    model_name='vision_agent',
    version_id='v1_20260610',
    model_state_dict=model.state_dict(),
    config=training_config.to_dict(),
    metrics=test_results,
    additional_files={
        'training_history.json': 'path/to/history.json'
    }
)
```

### Version Comparison

Compare two model versions:

```bash
python src/pipeline/model_versioning.py compare \
    --model-name vision_agent \
    --version-id1 v1_20260610_120000 \
    --version-id2 v2_20260610_150000
```

---

## ONNX Export

### Export Trained Model

Convert PyTorch model to ONNX format:

```bash
python src/pipeline/onnx_export.py \
    --model-path models/vision_agent/best_vision_agent.pth \
    --output-dir models/onnx \
    --optimize \
    --quantize
```

### Export Options

**Optimization:**
- Fuses operations
- Removes redundant nodes
- Improves inference speed

**Quantization:**
- Converts FP32 to INT8
- Reduces model size by ~75%
- Minimal accuracy loss (<1%)

### ONNX Inference

Use exported model for inference:

```python
from pipeline.onnx_export import ONNXInferenceEngine
import numpy as np

# Load ONNX model
engine = ONNXInferenceEngine('models/onnx/vision_agent.onnx')

# Prepare input (1, 1, 224, 224)
input_data = np.random.randn(1, 1, 224, 224).astype(np.float32)

# Run inference
output = engine.predict(input_data)
predicted_class = np.argmax(output)
confidence = np.max(output)

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.4f}")
```

### Benchmark ONNX Performance

```bash
python src/pipeline/onnx_export.py \
    --model-path models/vision_agent/best_vision_agent.pth \
    --benchmark
```

**Expected performance:**
- CPU inference: 20-50 ms per image
- GPU inference: 2-5 ms per image
- Batch inference: 100+ images/second

---

## Best Practices

### 1. Data Preparation

✅ **Do:**
- Verify data quality before training
- Check class balance
- Use stratified splits for imbalanced data
- Normalize images consistently

❌ **Don't:**
- Mix different image formats
- Use corrupted or low-quality images
- Ignore class imbalance

### 2. Training Strategy

✅ **Do:**
- Start with default hyperparameters
- Use early stopping to prevent overfitting
- Monitor both train and validation metrics
- Save checkpoints regularly

❌ **Don't:**
- Train for too many epochs without validation
- Use very high learning rates
- Ignore validation performance

### 3. Hyperparameter Tuning

✅ **Do:**
- Use Bayesian optimization for efficiency
- Start with 30-50 trials
- Focus on learning rate and batch size first
- Validate on held-out test set

❌ **Don't:**
- Tune on test set
- Use grid search for large spaces
- Ignore computational budget

### 4. Model Evaluation

✅ **Do:**
- Use cross-validation for robust estimates
- Report mean and standard deviation
- Analyze per-class performance
- Test on independent dataset

❌ **Don't:**
- Rely on single train/val split
- Only report overall accuracy
- Ignore class-specific metrics

### 5. Production Deployment

✅ **Do:**
- Export to ONNX for deployment
- Quantize for edge devices
- Version all models
- Monitor inference performance

❌ **Don't:**
- Deploy PyTorch models directly
- Skip model versioning
- Ignore inference latency

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce batch size: `--batch-size 16`
- Use gradient accumulation
- Enable mixed precision training
- Use CPU if GPU memory insufficient

#### 2. Poor Convergence

**Symptoms:**
- Loss not decreasing
- Validation accuracy stuck

**Solutions:**
- Reduce learning rate: `--lr 0.0001`
- Increase batch size
- Add data augmentation
- Check data quality

#### 3. Overfitting

**Symptoms:**
- High train accuracy, low validation accuracy
- Large gap between train/val loss

**Solutions:**
- Increase weight decay: `--weight-decay 1e-3`
- Add dropout
- Use more data augmentation
- Reduce model complexity

#### 4. Slow Training

**Symptoms:**
- Training takes too long
- Low GPU utilization

**Solutions:**
- Increase batch size
- Use more workers: `num_workers=8`
- Enable pin_memory
- Use mixed precision training

#### 5. ONNX Export Fails

**Symptoms:**
```
RuntimeError: ONNX export failed
```

**Solutions:**
- Update ONNX version
- Use supported opset version
- Simplify model architecture
- Check for unsupported operations

---

## Performance Optimization

### GPU Optimization

**Enable mixed precision training:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in train_loader:
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits:**
- 2-3x faster training
- 50% less memory usage
- Minimal accuracy impact

### Data Loading Optimization

**Optimize DataLoader:**
```python
train_loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,      # Use multiple workers
    pin_memory=True,    # Faster GPU transfer
    prefetch_factor=2   # Prefetch batches
)
```

### Inference Optimization

**Batch inference:**
```python
# Process multiple images at once
batch_size = 32
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    outputs = model(batch)
```

**ONNX quantization:**
```bash
python src/pipeline/onnx_export.py \
    --model-path models/best_model.pth \
    --quantize
```

---

## Advanced Topics

### Custom Loss Functions

Implement custom loss for class imbalance:

```python
import torch.nn as nn

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
    
    def forward(self, outputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weights)
        return ce_loss(outputs, targets)

# Use in training
class_weights = torch.tensor([1.0, 2.0, 2.0, 3.0])
criterion = WeightedCrossEntropyLoss(class_weights)
```

### Transfer Learning

Fine-tune from pretrained model:

```python
# Load pretrained model
model = AlzheimerVisionAgent(num_classes=4, pretrained=True)

# Freeze early layers
for param in model.model.layer1.parameters():
    param.requires_grad = False

# Train only later layers
optimizer = optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)
```

### Ensemble Methods

Combine multiple models:

```python
models = [
    load_model('models/fold_1_best.pth'),
    load_model('models/fold_2_best.pth'),
    load_model('models/fold_3_best.pth')
]

# Ensemble prediction
def ensemble_predict(models, input_data):
    predictions = []
    for model in models:
        output = model(input_data)
        predictions.append(output)
    
    # Average predictions
    ensemble_output = torch.mean(torch.stack(predictions), dim=0)
    return ensemble_output
```

---

## Appendix

### Command Reference

**Training:**
```bash
python src/pipeline/train_vision_agent.py --help
```

**Hyperparameter Tuning:**
```bash
python src/pipeline/hyperparameter_tuning.py --help
```

**Cross-Validation:**
```bash
python src/pipeline/cross_validation.py --help
```

**Model Versioning:**
```bash
python src/pipeline/model_versioning.py --help
```

**ONNX Export:**
```bash
python src/pipeline/onnx_export.py --help
```

### Configuration Templates

**Quick Training:**
```bash
python src/pipeline/train_vision_agent.py \
    --data-root data/oasis_raw \
    --epochs 30 \
    --batch-size 32
```

**Production Training:**
```bash
python src/pipeline/train_vision_agent.py \
    --data-root data/oasis_raw \
    --output-dir models/production \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --weight-decay 1e-4 \
    --lr-scheduler cosine \
    --device cuda
```

**Hyperparameter Search:**
```bash
python src/pipeline/hyperparameter_tuning.py \
    --data-root data/oasis_raw \
    --output-dir models/tuning \
    --n-trials 50 \
    --method bayesian \
    --train-best
```

---

## Support

For issues or questions:
- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Review [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- Open an issue on GitHub

---

**Document Version:** 1.0  
**Last Updated:** June 10, 2026  
**Maintained by:** OASIS Agentic Pipeline Team
