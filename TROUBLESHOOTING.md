# Troubleshooting Guide - OASIS Agentic Pipeline

**Last Updated:** June 10, 2026  
**Version:** 1.0

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Runtime Errors](#runtime-errors)
3. [Data Loading Problems](#data-loading-problems)
4. [Model Inference Issues](#model-inference-issues)
5. [Agent-Specific Problems](#agent-specific-problems)
6. [Dashboard Issues](#dashboard-issues)
7. [Performance Problems](#performance-problems)
8. [System-Specific Issues](#system-specific-issues)
9. [Common Error Messages](#common-error-messages)
10. [Getting Help](#getting-help)

---

## Installation Issues

### Problem: pip install fails with dependency conflicts

**Symptoms:**
```
ERROR: Cannot install package X because it conflicts with package Y
```

**Solutions:**

1. **Create fresh virtual environment:**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

2. **Install dependencies one by one:**
```bash
pip install torch torchvision
pip install transformers
pip install pandas numpy scikit-learn
pip install streamlit
```

3. **Use specific versions:**
```bash
pip install torch==2.12.0 torchvision==0.27.0
```

---

### Problem: CUDA not available after PyTorch installation

**Symptoms:**
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Solutions:**

1. **Verify CUDA installation:**
```bash
nvidia-smi
```

2. **Reinstall PyTorch with CUDA:**
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. **Check CUDA version compatibility:**
- PyTorch 2.12.0 supports CUDA 11.8 and 12.1
- Verify your NVIDIA driver version matches

4. **Set environment variable:**
```bash
# Windows
set CUDA_VISIBLE_DEVICES=0

# Linux/Mac
export CUDA_VISIBLE_DEVICES=0
```

---

### Problem: ModuleNotFoundError for installed packages

**Symptoms:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solutions:**

1. **Verify virtual environment is activated:**
```bash
# Check Python path
python -c "import sys; print(sys.executable)"
```

2. **Reinstall package:**
```bash
pip install --force-reinstall transformers
```

3. **Check Python version:**
```bash
python --version  # Should be 3.14+
```

---

## Runtime Errors

### Problem: Out of Memory (OOM) errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
```python
# In vision_agent.py
loader = DataLoader(dataset, batch_size=8, ...)  # Reduce from 32
```

2. **Use CPU instead of GPU:**
```python
device = torch.device("cpu")
```

3. **Clear GPU cache:**
```python
import torch
torch.cuda.empty_cache()
```

4. **Enable gradient checkpointing:**
```python
model.gradient_checkpointing_enable()
```

5. **Use mixed precision training:**
```python
from torch.cuda.amp import autocast
with autocast():
    output = model(input)
```

---

### Problem: Slow inference speed

**Symptoms:**
- Diagnosis takes > 10 seconds per patient
- Dashboard is unresponsive

**Solutions:**

1. **Use ONNX runtime:**
```python
from pipeline.onnx_inference.onnx_agent import ONNXMultimodalFusionAgent
agent = ONNXMultimodalFusionAgent("model.onnx")
```

2. **Batch processing:**
```python
# Process multiple patients at once
batch_tensor = torch.stack([img1, img2, img3])
outputs = model(batch_tensor)
```

3. **Disable gradient computation:**
```python
with torch.no_grad():
    output = model(input)
```

4. **Use GPU if available:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

---

## Data Loading Problems

### Problem: CSV file not found

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/oasis_raw/oasis_clinical_data.csv'
```

**Solutions:**

1. **Verify file path:**
```bash
# Windows
dir data\oasis_raw\oasis_clinical_data.csv

# Linux/Mac
ls data/oasis_raw/oasis_clinical_data.csv
```

2. **Check working directory:**
```python
import os
print(os.getcwd())  # Should be project root
```

3. **Use absolute paths:**
```python
csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "oasis_raw", "oasis_clinical_data.csv")
csv_path = os.path.abspath(csv_path)
```

4. **Generate synthetic data:**
```python
# The biomarker agent will auto-generate if file missing
agent = ClinicalBiomarkerAgent()
agent.ingest_and_process(csv_path)  # Creates synthetic data
```

---

### Problem: Image files not loading

**Symptoms:**
```
PIL.UnidentifiedImageError: cannot identify image file
```

**Solutions:**

1. **Verify image format:**
```bash
file data/oasis_raw/Non\ Demented/image.jpg
```

2. **Convert images to supported format:**
```python
from PIL import Image
img = Image.open("image.bmp")
img.save("image.jpg")
```

3. **Check file permissions:**
```bash
# Linux/Mac
chmod 644 data/oasis_raw/**/*.jpg
```

4. **Validate image integrity:**
```python
from PIL import Image
try:
    img = Image.open(path)
    img.verify()
except Exception as e:
    print(f"Corrupted image: {e}")
```

---

### Problem: Missing columns in CSV

**Symptoms:**
```
KeyError: 'MMSE'
```

**Solutions:**

1. **Check CSV headers:**
```python
import pandas as pd
df = pd.read_csv("data.csv")
print(df.columns.tolist())
```

2. **Required columns for clinical data:**
- Subject_ID
- M/F (Gender)
- Age
- Educ
- SES
- MMSE
- eTIV
- nWBV
- ASF

3. **Required columns for longitudinal data:**
- Subject ID (note the space)
- Visit
- MR Delay
- MMSE
- nWBV

4. **Add missing columns with defaults:**
```python
if 'MMSE' not in df.columns:
    df['MMSE'] = 25.0  # Default value
```

---

## Model Inference Issues

### Problem: Model weights not found

**Symptoms:**
```
FileNotFoundError: best_vision_agent.pth not found
```

**Solutions:**

1. **Train the model first:**
```bash
python src/agents/vision/vision_agent.py
```

2. **Use untrained model (for testing):**
```python
# Model will initialize with random weights
agent = AlzheimerVisionAgent(num_classes=4)
# Skip loading weights
```

3. **Download pre-trained weights:**
```bash
# If available from external source
wget https://example.com/weights.pth -O src/pipeline/onnx_inference/best_vision_agent.pth
```

4. **Check file path:**
```python
weights_path = os.path.join(workspace_root, "src", "pipeline", "onnx_inference", "best_vision_agent.pth")
print(f"Looking for weights at: {weights_path}")
```

---

### Problem: Incorrect prediction shapes

**Symptoms:**
```
RuntimeError: Expected tensor of shape [1, 1, 224, 224], got [1, 3, 224, 224]
```

**Solutions:**

1. **Convert to grayscale:**
```python
from PIL import Image
img = Image.open(path).convert('L')  # Force grayscale
```

2. **Check transform pipeline:**
```python
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure 1 channel
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

3. **Verify tensor dimensions:**
```python
print(f"Tensor shape: {img_tensor.shape}")
# Should be: torch.Size([1, 1, 224, 224])
```

---

### Problem: Low prediction confidence

**Symptoms:**
- All predictions below 60% confidence
- Ethicist agent constantly rejects diagnoses

**Solutions:**

1. **Train model longer:**
```python
# Increase epochs
for epoch in range(50):  # Instead of 10
    train_model()
```

2. **Check data quality:**
```python
# Verify images are not corrupted
# Ensure labels match image content
```

3. **Adjust confidence threshold:**
```python
ethicist = MedicalEthicistAgent(confidence_floor=50.0)  # Lower threshold
```

4. **Use ensemble methods:**
```python
# Average predictions from multiple models
pred1 = model1(input)
pred2 = model2(input)
final_pred = (pred1 + pred2) / 2
```

---

## Agent-Specific Problems

### Problem: RAG Agent fails to load model

**Symptoms:**
```
OSError: Can't load tokenizer for 'sentence-transformers/all-MiniLM-L6-v2'
```

**Solutions:**

1. **Check internet connection:**
```bash
ping huggingface.co
```

2. **Download model manually:**
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
```

3. **Use cached model:**
```bash
# Set cache directory
export TRANSFORMERS_CACHE=/path/to/cache
```

4. **Use alternative model:**
```python
agent = MedicalLibrarianAgent(model_name="distilbert-base-uncased")
```

---

### Problem: Temporal Agent returns empty metrics

**Symptoms:**
```python
metrics = {'status': 'No historical database loaded.', 'atrophy_velocity_pct': 0.0}
```

**Solutions:**

1. **Verify longitudinal CSV exists:**
```bash
ls data/oasis_raw/oasis_longitudinal.csv
```

2. **Check Subject ID format:**
```python
# Must match exactly: "OAS2_0001" not "OAS1_0001"
df = pd.read_csv("oasis_longitudinal.csv")
print(df['Subject ID'].unique())
```

3. **Ensure multiple visits:**
```python
# Patient needs at least 2 visits for progression tracking
patient_visits = df[df['Subject ID'] == 'OAS2_0001']
print(f"Visits: {len(patient_visits)}")
```

---

### Problem: Ethicist Agent always flags diagnoses

**Symptoms:**
- All diagnoses rejected
- "REJECTED: Sub-threshold confidence score"

**Solutions:**

1. **Lower confidence threshold:**
```python
ethicist = MedicalEthicistAgent(confidence_floor=50.0)
```

2. **Check input data quality:**
```python
# Verify MMSE and atrophy values are realistic
print(f"MMSE: {mmse_score}, Atrophy: {atrophy_velocity}")
```

3. **Review guardrail rules:**
```python
# In ethicist_agent.py, adjust rules if needed
if mmse_score >= 27.0 and predicted_class in ["Mild Dementia"]:
    # This rule may be too strict
```

---

## Dashboard Issues

### Problem: Streamlit dashboard won't start

**Symptoms:**
```
streamlit: command not found
```

**Solutions:**

1. **Install Streamlit:**
```bash
pip install streamlit
```

2. **Use full path:**
```bash
python -m streamlit run src/orchestrator/dashboard.py
```

3. **Check port availability:**
```bash
# Windows
netstat -ano | findstr :8501

# Linux/Mac
lsof -i :8501
```

4. **Use different port:**
```bash
streamlit run src/orchestrator/dashboard.py --server.port 8502
```

---

### Problem: Dashboard shows blank page

**Symptoms:**
- Browser loads but shows empty page
- No error messages

**Solutions:**

1. **Clear browser cache:**
```
Ctrl+Shift+Delete (Chrome/Edge)
Cmd+Shift+Delete (Safari)
```

2. **Check console for errors:**
```
F12 -> Console tab
```

3. **Restart Streamlit:**
```bash
# Kill process
Ctrl+C

# Restart
streamlit run src/orchestrator/dashboard.py
```

4. **Try different browser:**
- Chrome, Firefox, Edge, Safari

---

### Problem: Images not displaying in dashboard

**Symptoms:**
- Placeholder boxes instead of images
- "Image not found" errors

**Solutions:**

1. **Verify image paths:**
```python
print(f"Looking for images in: {test_folder}")
print(f"Files found: {os.listdir(test_folder)}")
```

2. **Check file permissions:**
```bash
# Linux/Mac
chmod 644 data/oasis_raw/**/*.jpg
```

3. **Use absolute paths:**
```python
image_path = os.path.abspath(os.path.join(ROOT_DIR, "data", "oasis_raw", "Non Demented", "image.jpg"))
```

---

## Performance Problems

### Problem: High memory usage

**Symptoms:**
- System becomes slow
- "MemoryError" exceptions

**Solutions:**

1. **Monitor memory usage:**
```python
import psutil
print(f"Memory: {psutil.virtual_memory().percent}%")
```

2. **Use lazy loading:**
```python
# Don't load all images at once
loader = DataLoader(dataset, batch_size=1, num_workers=0)
```

3. **Clear unused variables:**
```python
import gc
del large_tensor
gc.collect()
```

4. **Reduce model size:**
```python
# Use smaller backbone
model = models.resnet18()  # Instead of resnet50
```

---

### Problem: Long startup time

**Symptoms:**
- Dashboard takes > 30 seconds to load
- "Initializing..." message persists

**Solutions:**

1. **Use cached initialization:**
```python
@st.cache_resource
def load_model():
    return AlzheimerVisionAgent()
```

2. **Lazy load agents:**
```python
# Only initialize when needed
if user_clicks_diagnose:
    agent = load_agent()
```

3. **Pre-download models:**
```bash
# Download transformers models ahead of time
python -c "from transformers import AutoModel; AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"
```

---

## System-Specific Issues

### Windows Issues

#### Problem: Path separators causing errors

**Solution:**
```python
# Use os.path.join instead of hardcoded slashes
path = os.path.join("data", "oasis_raw", "file.csv")
```

#### Problem: PowerShell execution policy

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Problem: Long path names

**Solution:**
```
# Enable long paths in Windows 10+
# Run as Administrator:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

---

### Linux/Mac Issues

#### Problem: Permission denied errors

**Solution:**
```bash
chmod +x script.py
chmod -R 755 data/
```

#### Problem: Library not found

**Solution:**
```bash
# Install system dependencies
sudo apt-get install libgl1-mesa-glx  # For OpenCV
sudo apt-get install libjpeg-dev      # For PIL
```

---

### ARM64/Apple Silicon Issues

#### Problem: PyTorch not optimized for M1/M2

**Solution:**
```bash
# Install ARM64-optimized PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Problem: ONNX Runtime compatibility

**Solution:**
```bash
# Use CPU execution provider
pip install onnxruntime  # Not onnxruntime-gpu
```

---

## Common Error Messages

### "RuntimeError: Expected all tensors to be on the same device"

**Cause:** Model and input are on different devices (CPU vs GPU)

**Solution:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_tensor = input_tensor.to(device)
```

---

### "ValueError: Expected input batch_size (X) to match target batch_size (Y)"

**Cause:** Batch size mismatch between input and labels

**Solution:**
```python
# Ensure consistent batch sizes
assert inputs.shape[0] == labels.shape[0]
```

---

### "TypeError: can't convert cuda:0 device type tensor to numpy"

**Cause:** Trying to convert GPU tensor to numpy without moving to CPU

**Solution:**
```python
numpy_array = tensor.cpu().numpy()
```

---

### "AttributeError: 'NoneType' object has no attribute 'shape'"

**Cause:** Variable is None when it should be a tensor/array

**Solution:**
```python
if tensor is not None:
    print(tensor.shape)
else:
    print("Tensor is None!")
```

---

## Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide**
2. **Search existing issues on GitHub**
3. **Review error messages carefully**
4. **Try minimal reproducible example**

### Information to Provide

When reporting issues, include:

1. **System Information:**
```bash
python --version
pip list
nvidia-smi  # If using GPU
```

2. **Error Message:**
```
Full stack trace
```

3. **Code Snippet:**
```python
# Minimal code that reproduces the issue
```

4. **Steps to Reproduce:**
- Step 1: ...
- Step 2: ...
- Expected: ...
- Actual: ...

### Contact Channels

- **GitHub Issues:** [Project Repository]
- **Documentation:** [Project Wiki]
- **Email:** support@example.com
- **Discord:** [Community Server]

---

## Debugging Tips

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Use Python Debugger

```python
import pdb
pdb.set_trace()  # Breakpoint
```

### Profile Performance

```python
import cProfile
cProfile.run('model(input)')
```

### Check GPU Utilization

```bash
watch -n 1 nvidia-smi
```

### Monitor System Resources

```bash
# Linux
htop

# Windows
Task Manager (Ctrl+Shift+Esc)

# Mac
Activity Monitor
```

---

## Quick Reference

### Reset Everything

```bash
# Delete virtual environment
rm -rf .venv

# Create new environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Clear All Caches

```bash
# Pip cache
pip cache purge

# Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Transformers cache
rm -rf ~/.cache/huggingface
```

### Verify Installation

```python
import torch
import torchvision
import transformers
import pandas
import streamlit

print("All imports successful!")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```

---

*Last Updated: June 10, 2026*  
*For additional support, please refer to the project documentation or contact the development team.*
