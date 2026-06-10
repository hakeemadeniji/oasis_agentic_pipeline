# OASIS Agentic Pipeline - API Documentation

**Version:** 1.0.0  
**Last Updated:** June 10, 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Authentication](#authentication)
4. [REST API Endpoints](#rest-api-endpoints)
5. [WebSocket API](#websocket-api)
6. [Batch Processing](#batch-processing)
7. [Data Models](#data-models)
8. [Error Handling](#error-handling)
9. [Rate Limiting](#rate-limiting)
10. [Examples](#examples)

---

## Overview

The OASIS Agentic Pipeline API provides programmatic access to a multi-agent AI system for Alzheimer's disease diagnosis. The API supports:

- **REST API**: Synchronous diagnosis requests
- **WebSocket API**: Real-time streaming inference with progress updates
- **Batch Processing**: Efficient processing of multiple patients
- **Export Formats**: JSON, CSV for results

### Base URL

```
http://localhost:8000
```

### API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Getting Started

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the API server:
```bash
python src/api/main.py
```

Or using uvicorn directly:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Quick Test

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "cuda:0",
  "agents_loaded": true,
  "timestamp": "2026-06-10T16:30:00Z"
}
```

---

## Authentication

**Current Version:** No authentication required (development mode)

**Production Recommendation:** Implement OAuth2 or API key authentication before deployment.

---

## REST API Endpoints

### System Endpoints

#### GET /health

Health check endpoint to verify API status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "cuda:0",
  "agents_loaded": true,
  "timestamp": "2026-06-10T16:30:00Z"
}
```

#### GET /

Root endpoint with API information.

**Response:**
```json
{
  "message": "OASIS Agentic Pipeline API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

#### GET /stats

Get system statistics.

**Response:**
```json
{
  "total_patients": 100,
  "device": "cuda:0",
  "model_loaded": true,
  "agents_active": 6,
  "uptime": "N/A"
}
```

---

### Diagnosis Endpoints

#### POST /diagnose

Perform comprehensive multi-agent diagnosis for a single patient.

**Request Body:**
```json
{
  "patient_data": {
    "patient_id": "OAS2_0001",
    "age": 75.5,
    "mmse": 24.0,
    "gender": "F",
    "education": 12
  },
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB...",
  "longitudinal_id": "OAS2_0001"
}
```

**Parameters:**
- `patient_data` (required): Patient clinical information
  - `patient_id` (string, required): Unique patient identifier
  - `age` (float, required): Patient age in years (0-120)
  - `mmse` (float, required): MMSE cognitive score (0-30)
  - `gender` (string, optional): Patient gender (M/F)
  - `education` (integer, optional): Years of education
- `image_base64` (string, optional): Base64 encoded MRI image
- `longitudinal_id` (string, optional): ID for temporal analysis

**Response:**
```json
{
  "patient_id": "OAS2_0001",
  "timestamp": "2026-06-10T16:20:00Z",
  "vision_prediction": {
    "class": "Very Mild Dementia",
    "confidence": 87.5,
    "probabilities": [0.05, 0.875, 0.05, 0.025]
  },
  "biomarker_analysis": {
    "age_risk": "elevated",
    "mmse_category": "mild_impairment"
  },
  "temporal_analysis": {
    "trend": "Typical Age-Related Neuro-Degradation",
    "atrophy_velocity": 0.45,
    "visits_tracked": 3,
    "years_monitored": 2.5
  },
  "rag_context": [
    "Clinical guideline text..."
  ],
  "explainability": {
    "heatmap_available": true,
    "peak_activation": [112, 98],
    "heatmap_shape": [224, 224]
  },
  "ethics_audit": {
    "approved": true,
    "message": "All safety checks passed"
  },
  "final_diagnosis": "Very Mild Dementia",
  "confidence": 87.5,
  "approved": true
}
```

**Status Codes:**
- `200`: Success
- `400`: Invalid request
- `500`: Internal server error
- `503`: Service unavailable (CMO not initialized)

**Example:**
```python
import requests
import base64

# Load and encode image
with open("mri_scan.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8000/diagnose",
    json={
        "patient_data": {
            "patient_id": "OAS2_0001",
            "age": 75.5,
            "mmse": 24.0
        },
        "image_base64": image_base64
    }
)

result = response.json()
print(f"Diagnosis: {result['final_diagnosis']}")
print(f"Confidence: {result['confidence']}%")
```

---

#### POST /diagnose/batch

Perform batch diagnosis for multiple patients.

**Request Body:**
```json
{
  "requests": [
    {
      "patient_data": {
        "patient_id": "OAS2_0001",
        "age": 75.5,
        "mmse": 24.0
      },
      "image_base64": "..."
    },
    {
      "patient_data": {
        "patient_id": "OAS2_0002",
        "age": 68.0,
        "mmse": 28.0
      }
    }
  ]
}
```

**Parameters:**
- `requests` (array, required): List of diagnosis requests (max 100)

**Response:**
```json
{
  "job_id": "batch_20260610_162000",
  "status": "processing",
  "total_requests": 2,
  "message": "Batch processing started"
}
```

**Status Codes:**
- `200`: Batch job started
- `400`: Invalid request (e.g., too many requests)
- `503`: Service unavailable

**Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/diagnose/batch",
    json={
        "requests": [
            {
                "patient_data": {
                    "patient_id": "OAS2_0001",
                    "age": 75.5,
                    "mmse": 24.0
                }
            },
            {
                "patient_data": {
                    "patient_id": "OAS2_0002",
                    "age": 68.0,
                    "mmse": 28.0
                }
            }
        ]
    }
)

job_id = response.json()["job_id"]
print(f"Batch job started: {job_id}")
```

---

### Upload Endpoints

#### POST /upload/image

Upload MRI image and get base64 encoding for diagnosis.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `file` (image file)

**Response:**
```json
{
  "filename": "mri_scan.png",
  "size": 45678,
  "format": "PNG",
  "dimensions": [224, 224],
  "base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB..."
}
```

**Example:**
```python
import requests

with open("mri_scan.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload/image",
        files={"file": f}
    )

image_data = response.json()
base64_image = image_data["base64"]
```

---

### Patient Endpoints

#### GET /patients/{patient_id}

Get patient data by ID.

**Parameters:**
- `patient_id` (string, required): Patient identifier

**Response:**
```json
{
  "patient_id": "OAS2_0001",
  "data": {
    "Subject_ID": "OAS2_0001",
    "Age": 75.5,
    "MMSE": 24.0,
    "Gender": "F",
    "Education": 12
  }
}
```

**Status Codes:**
- `200`: Success
- `404`: Patient not found
- `503`: Service unavailable

---

#### GET /patients

List all patients with pagination.

**Query Parameters:**
- `limit` (integer, optional): Number of results per page (1-100, default: 10)
- `offset` (integer, optional): Starting index (default: 0)

**Response:**
```json
{
  "total": 100,
  "limit": 10,
  "offset": 0,
  "patients": [
    {
      "Subject_ID": "OAS2_0001",
      "Age": 75.5,
      "MMSE": 24.0
    }
  ]
}
```

**Example:**
```python
import requests

# Get first 20 patients
response = requests.get(
    "http://localhost:8000/patients",
    params={"limit": 20, "offset": 0}
)

patients = response.json()["patients"]
```

---

### Model Endpoints

#### GET /models/info

Get information about loaded models.

**Response:**
```json
{
  "vision_agent": {
    "architecture": "ResNet18",
    "input_shape": [1, 1, 224, 224],
    "num_classes": 4,
    "classes": [
      "Non Demented",
      "Very Mild Dementia",
      "Mild Dementia",
      "Moderate Dementia"
    ]
  },
  "device": "cuda:0",
  "agents": {
    "vision": "AlzheimerVisionAgent",
    "biomarker": "BiomarkerAgent",
    "rag": "RAGAgent",
    "explainer": "ExplainerAgent",
    "temporal": "TemporalAnalyst",
    "ethicist": "EthicistAgent"
  }
}
```

---

### Export Endpoints

#### GET /export/{job_id}

Export diagnosis results in specified format.

**Parameters:**
- `job_id` (string, required): Batch job ID or patient ID
- `format` (string, optional): Export format (`json` or `csv`, default: `json`)

**Response:**
- Content-Type: `application/json` or `text/csv`
- File download

**Example:**
```python
import requests

# Export as JSON
response = requests.get(
    "http://localhost:8000/export/batch_20260610_162000",
    params={"format": "json"}
)

with open("results.json", "wb") as f:
    f.write(response.content)

# Export as CSV
response = requests.get(
    "http://localhost:8000/export/batch_20260610_162000",
    params={"format": "csv"}
)

with open("results.csv", "wb") as f:
    f.write(response.content)
```

---

## WebSocket API

### Connection

Connect to WebSocket endpoint for real-time streaming inference.

**Endpoint:** `ws://localhost:8000/ws/{client_id}`

**Parameters:**
- `client_id` (string, required): Unique client identifier

### Message Types

#### 1. Diagnose Request

**Send:**
```json
{
  "type": "diagnose",
  "patient_data": {
    "patient_id": "OAS2_0001",
    "age": 75.5,
    "mmse": 24.0
  },
  "image_base64": "..."
}
```

**Receive (Progress Updates):**
```json
{
  "type": "progress",
  "stage": "vision_analysis",
  "progress": 0.3,
  "message": "Running Vision Agent analysis...",
  "timestamp": "2026-06-10T16:30:00Z"
}
```

**Receive (Agent Results):**
```json
{
  "type": "result",
  "agent": "vision",
  "data": {
    "class": "Very Mild Dementia",
    "confidence": 87.5
  }
}
```

**Receive (Final Result):**
```json
{
  "type": "final",
  "data": {
    "patient_id": "OAS2_0001",
    "diagnosis": "Very Mild Dementia",
    "confidence": 87.5,
    "approved": true,
    "timestamp": "2026-06-10T16:30:05Z"
  }
}
```

#### 2. Batch Request

**Send:**
```json
{
  "type": "batch",
  "patients": [
    {
      "patient_data": {
        "patient_id": "OAS2_0001",
        "age": 75.5,
        "mmse": 24.0
      }
    }
  ]
}
```

**Receive (Batch Progress):**
```json
{
  "type": "batch_progress",
  "current": 1,
  "total": 10,
  "progress": 0.1,
  "patient_id": "OAS2_0001"
}
```

#### 3. Heartbeat

**Send:**
```json
{
  "type": "ping"
}
```

**Receive:**
```json
{
  "type": "pong"
}
```

### Python Example

```python
import asyncio
import websockets
import json
import base64

async def diagnose_patient():
    uri = "ws://localhost:8000/ws/client_123"
    
    async with websockets.connect(uri) as websocket:
        # Load image
        with open("mri_scan.png", "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
        
        # Send diagnosis request
        request = {
            "type": "diagnose",
            "patient_data": {
                "patient_id": "OAS2_0001",
                "age": 75.5,
                "mmse": 24.0
            },
            "image_base64": image_base64
        }
        
        await websocket.send(json.dumps(request))
        
        # Receive results
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            
            if data["type"] == "progress":
                print(f"Progress: {data['message']} ({data['progress']*100:.0f}%)")
            
            elif data["type"] == "result":
                print(f"Agent {data['agent']}: {data['data']}")
            
            elif data["type"] == "final":
                print(f"Final diagnosis: {data['data']['diagnosis']}")
                break
            
            elif data["type"] == "error":
                print(f"Error: {data['message']}")
                break

asyncio.run(diagnose_patient())
```

### JavaScript Example

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/client_123');

ws.onopen = () => {
    console.log('Connected to WebSocket');
    
    // Send diagnosis request
    ws.send(JSON.stringify({
        type: 'diagnose',
        patient_data: {
            patient_id: 'OAS2_0001',
            age: 75.5,
            mmse: 24.0
        },
        image_base64: '...'
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'progress') {
        console.log(`Progress: ${data.message} (${data.progress * 100}%)`);
    } else if (data.type === 'result') {
        console.log(`Agent ${data.agent}:`, data.data);
    } else if (data.type === 'final') {
        console.log('Final diagnosis:', data.data.diagnosis);
        ws.close();
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};
```

---

## Batch Processing

### Command Line Interface

Process multiple patients from CSV file:

```bash
python src/api/batch_processor.py \
    --csv data/patients.csv \
    --image-dir data/mri_images \
    --parallel \
    --workers 4 \
    --export-csv
```

**Arguments:**
- `--csv`: Path to CSV file with patient data (required)
- `--image-dir`: Directory containing MRI images (optional)
- `--output`: Output file path for results (optional)
- `--parallel`: Use parallel processing (default: true)
- `--workers`: Number of parallel workers (default: 4)
- `--export-csv`: Export results to CSV (optional)

### CSV Format

**Input CSV:**
```csv
patient_id,age,mmse,gender,education,image_filename,longitudinal_id
OAS2_0001,75.5,24.0,F,12,scan_001.png,OAS2_0001
OAS2_0002,68.0,28.0,M,16,scan_002.png,OAS2_0002
```

**Output CSV:**
```csv
patient_id,timestamp,diagnosis,confidence,approved,vision_class,vision_confidence,temporal_trend,atrophy_velocity,ethics_approved,ethics_message
OAS2_0001,2026-06-10T16:30:00Z,Very Mild Dementia,87.5,true,Very Mild Dementia,87.5,Typical Age-Related Neuro-Degradation,0.45,true,All safety checks passed
```

### Python API

```python
from src.api.batch_processor import BatchProcessor

# Initialize processor
processor = BatchProcessor(
    workspace_root=".",
    max_workers=4,
    use_gpu=True
)

# Process CSV file
results = processor.process_csv_file(
    csv_path="data/patients.csv",
    image_dir="data/mri_images",
    parallel=True,
    save_results=True
)

# Export to CSV
processor.export_results_to_csv(
    results['results'],
    "output/batch_results.csv"
)

print(f"Processed {results['summary']['total_patients']} patients")
print(f"Throughput: {results['summary']['throughput_patients_per_hour']:.1f} patients/hour")
```

---

## Data Models

### PatientData

```python
{
  "patient_id": str,      # Required: Unique identifier
  "age": float,           # Required: 0-120 years
  "mmse": float,          # Required: 0-30 score
  "gender": str,          # Optional: "M" or "F"
  "education": int        # Optional: Years of education
}
```

### DiagnosisRequest

```python
{
  "patient_data": PatientData,     # Required
  "image_base64": str,             # Optional: Base64 encoded image
  "longitudinal_id": str           # Optional: For temporal analysis
}
```

### DiagnosisResponse

```python
{
  "patient_id": str,
  "timestamp": str,                # ISO 8601 format
  "vision_prediction": {
    "class": str,
    "confidence": float,
    "probabilities": list[float]
  },
  "biomarker_analysis": dict,
  "temporal_analysis": {
    "trend": str,
    "atrophy_velocity": float,
    "visits_tracked": int,
    "years_monitored": float
  },
  "rag_context": list[str],
  "explainability": {
    "heatmap_available": bool,
    "peak_activation": list[int],
    "heatmap_shape": list[int]
  },
  "ethics_audit": {
    "approved": bool,
    "message": str
  },
  "final_diagnosis": str,
  "confidence": float,
  "approved": bool
}
```

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message description"
}
```

### Common Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 400 | Bad Request | Check request format and parameters |
| 404 | Not Found | Verify resource ID exists |
| 500 | Internal Server Error | Check server logs |
| 503 | Service Unavailable | Wait for service initialization |

### Error Examples

**Invalid Patient Data:**
```json
{
  "detail": "age: ensure this value is less than or equal to 120"
}
```

**Patient Not Found:**
```json
{
  "detail": "Patient not found"
}
```

**Service Unavailable:**
```json
{
  "detail": "CMO not initialized"
}
```

---

## Rate Limiting

**Current Version:** No rate limiting (development mode)

**Production Recommendation:**
- Implement rate limiting per client/API key
- Suggested limits:
  - Single diagnosis: 60 requests/minute
  - Batch processing: 10 requests/minute
  - WebSocket connections: 5 concurrent per client

---

## Examples

### Complete Workflow Example

```python
import requests
import base64
import time

# 1. Check API health
health = requests.get("http://localhost:8000/health").json()
print(f"API Status: {health['status']}")

# 2. Upload image
with open("mri_scan.png", "rb") as f:
    upload_response = requests.post(
        "http://localhost:8000/upload/image",
        files={"file": f}
    )
image_base64 = upload_response.json()["base64"]

# 3. Get patient data
patient = requests.get(
    "http://localhost:8000/patients/OAS2_0001"
).json()

# 4. Perform diagnosis
diagnosis = requests.post(
    "http://localhost:8000/diagnose",
    json={
        "patient_data": {
            "patient_id": patient["patient_id"],
            "age": patient["data"]["Age"],
            "mmse": patient["data"]["MMSE"]
        },
        "image_base64": image_base64,
        "longitudinal_id": patient["patient_id"]
    }
).json()

# 5. Display results
print(f"\nDiagnosis Results:")
print(f"Patient: {diagnosis['patient_id']}")
print(f"Diagnosis: {diagnosis['final_diagnosis']}")
print(f"Confidence: {diagnosis['confidence']:.2f}%")
print(f"Approved: {diagnosis['approved']}")
print(f"\nVision Agent: {diagnosis['vision_prediction']['class']}")
print(f"Temporal Trend: {diagnosis['temporal_analysis']['trend']}")
print(f"Ethics Audit: {diagnosis['ethics_audit']['message']}")
```

### Batch Processing Example

```python
import requests
import time

# Submit batch job
batch_response = requests.post(
    "http://localhost:8000/diagnose/batch",
    json={
        "requests": [
            {
                "patient_data": {
                    "patient_id": f"OAS2_{i:04d}",
                    "age": 70 + i,
                    "mmse": 25 - i * 0.5
                }
            }
            for i in range(10)
        ]
    }
).json()

job_id = batch_response["job_id"]
print(f"Batch job started: {job_id}")

# Wait for completion
time.sleep(30)

# Download results
results = requests.get(
    f"http://localhost:8000/export/{job_id}",
    params={"format": "json"}
)

with open(f"{job_id}.json", "wb") as f:
    f.write(results.content)

print(f"Results saved to {job_id}.json")
```

---

## Support

For issues, questions, or contributions:

- **GitHub**: https://github.com/oasis-pipeline
- **Issues**: https://github.com/oasis-pipeline/issues
- **Documentation**: http://localhost:8000/docs

---

**Last Updated:** June 10, 2026  
**Version:** 1.0.0  
**Maintained by:** OASIS Development Team