# Python Dependencies - OASIS Agentic Pipeline

**Generated:** June 10, 2026  
**Python Version Required:** 3.14+

---

## Installation Instructions

### Quick Install
```bash
pip install -r requirements.txt
```

### GPU Support (CUDA)
```bash
# Ensure CUDA toolkit is installed first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### CPU-Only Installation
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## Core Dependencies

### Deep Learning Framework
| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.12.0 | Core PyTorch framework for neural networks |
| torchvision | 0.27.0 | Computer vision models and transforms |

### Transformers & NLP
| Package | Version | Purpose |
|---------|---------|---------|
| transformers | 5.10.2 | HuggingFace transformers for RAG agent |
| tokenizers | 0.22.2 | Fast tokenization for NLP models |
| huggingface_hub | 1.18.0 | Model hub integration |
| safetensors | 0.8.0 | Safe tensor serialization |

### Data Processing
| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 2.4.6 | Numerical computing foundation |
| pandas | 3.0.3 | Tabular data manipulation |
| scipy | 1.17.1 | Scientific computing algorithms |
| scikit-learn | 1.9.0 | Machine learning utilities |
| scikit-image | 0.26.0 | Image processing algorithms |

### Medical Imaging
| Package | Version | Purpose |
|---------|---------|---------|
| nibabel | 5.4.2 | Neuroimaging file format support |
| ImageIO | 2.37.3 | Image reading/writing |
| pillow | 12.2.0 | Python Imaging Library |
| tifffile | 2026.6.1 | TIFF file handling |

### ONNX Runtime
| Package | Version | Purpose |
|---------|---------|---------|
| onnx | 1.21.0 | ONNX model format |
| onnx-ir | 0.2.1 | ONNX intermediate representation |
| onnxruntime | 1.26.0 | ONNX inference engine |
| onnxscript | 0.7.0 | ONNX scripting tools |
| flatbuffers | 25.12.19 | Serialization library |
| protobuf | 7.35.0 | Protocol buffers |

### Web Framework
| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.58.0 | Dashboard web interface |
| starlette | 1.2.1 | ASGI framework |
| uvicorn | 0.49.0 | ASGI server |
| python-multipart | 0.0.32 | Multipart form data parser |
| httptools | 0.8.0 | HTTP parsing |
| websockets | 16.0 | WebSocket support |
| watchdog | 6.0.0 | File system monitoring |

### HTTP & Networking
| Package | Version | Purpose |
|---------|---------|---------|
| httpx | 0.28.1 | Modern HTTP client |
| httpcore | 1.0.9 | HTTP core functionality |
| h11 | 0.16.0 | HTTP/1.1 protocol |
| requests | 2.34.2 | HTTP library |
| urllib3 | 2.7.0 | HTTP client |
| certifi | 2026.5.20 | SSL certificates |
| idna | 3.18 | Internationalized domain names |
| charset-normalizer | 3.4.7 | Character encoding detection |
| anyio | 4.13.0 | Async I/O abstraction |

### Visualization
| Package | Version | Purpose |
|---------|---------|---------|
| altair | 6.2.1 | Declarative visualization |
| pydeck | 0.9.2 | Deck.gl visualization |
| rich | 15.0.0 | Rich terminal output |
| Pygments | 2.20.0 | Syntax highlighting |
| markdown-it-py | 4.2.0 | Markdown parsing |
| mdurl | 0.1.2 | Markdown URL utilities |

### Utilities
| Package | Version | Purpose |
|---------|---------|---------|
| click | 8.4.1 | CLI creation framework |
| typer | 0.25.1 | Modern CLI framework |
| tqdm | 4.68.2 | Progress bars |
| joblib | 1.5.3 | Parallel computing |
| filelock | 3.29.1 | File locking |
| fsspec | 2026.4.0 | File system abstraction |
| tenacity | 9.1.4 | Retry logic |
| cachetools | 7.1.4 | Caching utilities |
| blinker | 1.9.0 | Signal/event system |
| colorama | 0.4.6 | Cross-platform colored terminal |
| shellingham | 1.5.4 | Shell detection |
| threadpoolctl | 3.6.0 | Thread pool control |

### Configuration
| Package | Version | Purpose |
|---------|---------|---------|
| PyYAML | 6.0.3 | YAML parser |
| toml | 0.10.2 | TOML parser |
| packaging | 26.2 | Package version handling |

### Data Validation
| Package | Version | Purpose |
|---------|---------|---------|
| jsonschema | 4.26.0 | JSON schema validation |
| jsonschema-specifications | 2025.9.1 | JSON schema specs |
| referencing | 0.37.0 | JSON reference resolution |
| rpds-py | 2026.5.1 | Persistent data structures |
| attrs | 26.1.0 | Classes without boilerplate |

### Template Engine
| Package | Version | Purpose |
|---------|---------|---------|
| Jinja2 | 3.1.6 | Template engine |
| MarkupSafe | 3.0.3 | Safe string handling |
| itsdangerous | 2.2.0 | Secure data signing |

### Version Control
| Package | Version | Purpose |
|---------|---------|---------|
| GitPython | 3.1.50 | Git repository interaction |
| gitdb | 4.0.12 | Git object database |
| smmap | 5.0.3 | Memory-mapped file support |

### Date & Time
| Package | Version | Purpose |
|---------|---------|---------|
| python-dateutil | 2.9.0.post0 | Date/time extensions |
| tzdata | 2026.2 | Timezone database |

### Mathematical
| Package | Version | Purpose |
|---------|---------|---------|
| sympy | 1.14.0 | Symbolic mathematics |
| mpmath | 1.3.0 | Arbitrary-precision arithmetic |
| ml_dtypes | 0.5.4 | Machine learning data types |
| networkx | 3.6.1 | Graph algorithms |
| lazy-loader | 0.5 | Lazy module loading |
| narwhals | 2.22.1 | DataFrame abstraction |

### Data Formats
| Package | Version | Purpose |
|---------|---------|---------|
| pyarrow | 24.0.0 | Apache Arrow format |

### Miscellaneous
| Package | Version | Purpose |
|---------|---------|---------|
| annotated-doc | 0.0.4 | Documentation annotations |
| regex | 2026.5.9 | Regular expressions |
| six | 1.17.0 | Python 2/3 compatibility |
| typing_extensions | 4.15.0 | Typing backports |

### Build Tools
| Package | Version | Purpose |
|---------|---------|---------|
| setuptools | 81.0.0 | Package building |
| wheel | 0.47.0 | Wheel package format |

---

## System Requirements

### Minimum
- **Python:** 3.14+
- **RAM:** 8GB
- **Storage:** 10GB for dependencies
- **CPU:** 4 cores

### Recommended
- **Python:** 3.14+
- **RAM:** 16GB+
- **Storage:** 50GB+ SSD
- **CPU:** 8+ cores
- **GPU:** NVIDIA with 6GB+ VRAM (CUDA support)

---

## Creating requirements.txt

To generate a `requirements.txt` file from this environment:

```bash
pip freeze > requirements.txt
```

Or create manually with core dependencies:

```txt
torch==2.12.0
torchvision==0.27.0
transformers==5.10.2
numpy==2.4.6
pandas==3.0.3
scikit-learn==1.9.0
nibabel==5.4.2
pillow==12.2.0
streamlit==1.58.0
onnxruntime==1.26.0
```

---

## Troubleshooting

### CUDA Not Detected
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
```bash
# Reduce batch size in training scripts
# Use CPU-only mode if GPU memory insufficient
export CUDA_VISIBLE_DEVICES=""
```

### Import Errors
```bash
# Reinstall specific package
pip install --force-reinstall <package-name>

# Clear pip cache
pip cache purge
```

---

*Last Updated: June 10, 2026*
