# 🚀 Quick Start Guide

Get the OASIS Agentic Pipeline running in **5 minutes**.

---

## ⚡ Prerequisites

- **Python 3.14+** ([Download](https://www.python.org/downloads/))
- **8GB RAM** minimum
- **Internet connection** (for downloading dependencies)

---

## 📦 Installation (3 Steps)

### 1. Clone & Navigate
```bash
git clone https://github.com/hakeemadeniji/oasis-agentic-pipeline.git
cd oasis-agentic-pipeline
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Installation complete!** ✅

---

## 🎯 Run Your First Diagnosis

### Option A: Web Dashboard (Easiest)
```bash
streamlit run src/orchestrator/dashboard.py
```
Then open your browser to: **http://localhost:8501**

### Option B: Terminal Interface
```bash
python src/orchestrator/terminal_cmo.py
```

### Option C: Python Script
Create `test_diagnosis.py`:
```python
from orchestrator.chief_medical_officer import AdvancedChiefMedicalOfficer

# Initialize the system
cmo = AdvancedChiefMedicalOfficer(workspace_root=".")

# Run diagnosis on first patient
cmo.execute_comprehensive_diagnosis(
    patient_idx=0,
    image_path="data/oasis_raw/Non Demented/26 - 1.jpg",
    mock_subject_id="OAS2_0001"
)
```

Run it:
```bash
python test_diagnosis.py
```

---

## 📊 What You'll See

The system will output:

1. **Vision Analysis**: MRI classification with confidence scores
2. **Clinical Assessment**: Biomarker analysis (MMSE, brain volume)
3. **Explainability**: Grad-CAM heatmap showing decision regions
4. **Temporal Trends**: Disease progression velocity (if longitudinal data exists)
5. **Medical Evidence**: RAG-retrieved literature supporting the diagnosis
6. **Ethical Validation**: Compliance checks and confidence thresholds
7. **Final Diagnosis**: Integrated multi-agent consensus

---

## 🔧 Quick Troubleshooting

### Problem: "No module named 'torch'"
**Solution:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Problem: "CUDA not available" (GPU users)
**Solution:**
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Problem: "FileNotFoundError: oasis_clinical_data.csv"
**Solution:** The system will auto-generate synthetic data. Just proceed with the diagnosis.

### Problem: Streamlit won't start
**Solution:**
```bash
pip install --upgrade streamlit
streamlit run src/orchestrator/dashboard.py --server.port 8502
```

---

## 📁 Quick File Reference

```
Key Files You'll Use:
├── src/orchestrator/dashboard.py          # Web interface
├── src/orchestrator/terminal_cmo.py       # CLI interface
├── src/orchestrator/chief_medical_officer.py  # Main orchestrator
├── data/oasis_raw/                        # Place your MRI images here
└── requirements.txt                       # All dependencies
```

---

## 🎓 Next Steps

1. **Explore the Dashboard**: Upload your own MRI images
2. **Read the Full README**: [README.md](README.md) for architecture details
3. **Check API Docs**: [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for integration
4. **Review Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues

---

## 💡 Pro Tips

- **First run is slower**: Model initialization takes ~30 seconds
- **Use GPU if available**: 3-5x faster inference
- **Batch processing**: Process multiple patients at once for efficiency
- **HITL Queue**: Check `data/active_learning.db` for flagged cases

---

## 🆘 Need Help?

- **Detailed Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **API Documentation**: [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- **Report Issues**: [GitHub Issues](https://github.com/hakeemadeniji/oasis-agentic-pipeline/issues)

---

## ✅ Verification Checklist

After installation, verify everything works:

```bash
# Check Python version
python --version  # Should be 3.14+

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check all agents load
python -c "from orchestrator.chief_medical_officer import AdvancedChiefMedicalOfficer; print('✅ All agents loaded')"

# Test Streamlit
streamlit --version
```

All checks passed? **You're ready to go!** 🎉

---

**Total Setup Time: ~5 minutes** ⏱️

*Last Updated: June 10, 2026*
