# OASIS Agentic Pipeline API Documentation

Complete API reference for the OASIS Agentic Pipeline REST API.

## Base URL

```
Development: http://localhost:8000
Production: https://api.oasis-pipeline.com
```

## Authentication

The API uses OAuth2 password flow for authentication. Include the access token in the Authorization header:

```
Authorization: Bearer <your_access_token>
```

### Get Access Token

**Endpoint:** `POST /api/auth/token`

**Request:**
```bash
curl -X POST "http://localhost:8000/api/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=clinician&password=password123"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "user": {
    "username": "clinician",
    "email": "clinician@hospital.com",
    "role": "clinician",
    "full_name": "Dr. Smith"
  }
}
```

### Refresh Token

**Endpoint:** `POST /api/auth/refresh`

**Request:**
```bash
curl -X POST "http://localhost:8000/api/auth/refresh" \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "your_refresh_token"}'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

## API Endpoints

### Health Check

**Endpoint:** `GET /health`

**Description:** Check API health status

**Authentication:** None required

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "cuda",
  "agents_loaded": true,
  "timestamp": "2026-06-17T12:00:00Z"
}
```

### Single Patient Diagnosis

**Endpoint:** `POST /diagnose`

**Description:** Perform comprehensive multi-agent diagnosis for a single patient

**Authentication:** Optional (public access with rate limiting)

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
  "image_base64": "base64_encoded_image_data",
  "longitudinal_id": "OAS2_0001"
}
```

**Response:**
```json
{
  "patient_id": "OAS2_0001",
  "timestamp": "2026-06-17T12:00:00Z",
  "vision_prediction": {
    "predicted_class": "Very Mild Dementia",
    "confidence": 87.5,
    "probabilities": {
      "Non Demented": 5.2,
      "Very Mild Dementia": 87.5,
      "Mild Dementia": 6.8,
      "Moderate Dementia": 0.5
    }
  },
  "biomarker_analysis": {
    "mmse_score": 24.0,
    "mmse_interpretation": "Mild cognitive impairment",
    "age_adjusted_mmse": 22.5,
    "risk_factors": ["age", "education"]
  },
  "temporal_analysis": {
    "atrophy_velocity_pct": 0.8,
    "clinical_trend": "stable",
    "trajectory": "slow_decline"
  },
  "rag_context": [
    "MMSE scores between 20-26 indicate mild cognitive impairment.",
    "Patients with MMSE <24 typically show hippocampal atrophy."
  ],
  "explainability": {
    "heatmap_available": true,
    "attention_regions": ["hippocampus", "entorhinal_cortex"]
  },
  "ethics_audit": {
    "flagged": false,
    "message": "VERIFIED",
    "confidence_adequate": true,
    "clinical_consistency": true
  },
  "final_diagnosis": {
    "diagnosis": "Very Mild Dementia",
    "confidence": 87.5,
    "approved": true,
    "tier": "cheap"
  },
  "confidence": 87.5,
  "approved": true
}
```

### Batch Diagnosis

**Endpoint:** `POST /diagnose/batch`

**Description:** Perform batch diagnosis for multiple patients

**Authentication:** Required (Clinician role or higher)

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
      "image_base64": "base64_encoded_image_1"
    },
    {
      "patient_data": {
        "patient_id": "OAS2_0002",
        "age": 78.0,
        "mmse": 22.0
      },
      "image_base64": "base64_encoded_image_2"
    }
  ],
  "background_processing": false
}
```

**Response:**
```json
{
  "job_id": "batch_job_12345",
  "status": "completed",
  "total_patients": 2,
  "completed_patients": 2,
  "failed_patients": 0,
  "results": [
    {
      "patient_id": "OAS2_0001",
      "diagnosis": "Very Mild Dementia",
      "confidence": 87.5
    },
    {
      "patient_id": "OAS2_0002",
      "diagnosis": "Mild Dementia",
      "confidence": 82.3
    }
  ],
  "processing_time_seconds": 15.2
}
```

### Image Upload

**Endpoint:** `POST /upload`

**Description:** Upload MRI image for processing

**Authentication:** Optional

**Request:** `multipart/form-data`

- `file`: Image file (JPEG, PNG, TIFF)
- `patient_id`: Patient identifier

**Response:**
```json
{
  "success": true,
  "image_id": "img_12345",
  "patient_id": "OAS2_0001",
  "filename": "mri_scan.jpg",
  "size_bytes": 2048576,
  "format": "JPEG",
  "uploaded_at": "2026-06-17T12:00:00Z"
}
```

### Get Heatmap

**Endpoint:** `GET /heatmap/{image_id}`

**Description:** Generate Grad-CAM heatmap for image

**Authentication:** Optional

**Parameters:**
- `image_id`: Image identifier
- `target_class`: Target class for heatmap (optional)

**Response:** `image/png`

### Get Patient Data

**Endpoint:** `GET /patients/{patient_id}`

**Description:** Retrieve patient clinical data

**Authentication:** Required (Viewer role or higher)

**Response:**
```json
{
  "patient_id": "OAS2_0001",
  "age": 75.5,
  "gender": "F",
  "mmse": 24.0,
  "education": 12,
  "ses": 3,
  "etiv": 1450.0,
  "nwbv": 0.72,
  "asf": 1.15
}
```

### Research/Cure Analysis

**Endpoint:** `GET /research`

**Description:** Run cure-research analysis and therapeutic hypothesis generation

**Authentication:** Required (Researcher role or higher)

**Response:**
```json
{
  "deterministic_findings": [
    {
      "biomarker": "MMSE",
      "finding": "Strong correlation with disease progression",
      "p_value": 0.001,
      "effect_size": 0.65
    }
  ],
  "therapeutic_hypotheses": [
    {
      "mechanism": "amyloid_clearance",
      "target": "BACE1",
      "rationale": "Reduces amyloid-beta production",
      "feasibility": "high"
    }
  ],
  "literature_references": [
    "Smith et al. (2024), Nature Medicine"
  ]
}
```

### System Status

**Endpoint:** `GET /status`

**Description:** Get detailed system status and metrics

**Authentication:** Required (Admin role)

**Response:**
```json
{
  "system": {
    "status": "operational",
    "uptime_seconds": 86400,
    "version": "1.0.0"
  },
  "agents": {
    "vision": "loaded",
    "biomarker": "loaded",
    "temporal": "loaded",
    "rag": "loaded",
    "ethicist": "loaded",
    "reasoner": "loaded"
  },
  "performance": {
    "avg_response_time_ms": 1250,
    "requests_per_minute": 45,
    "cache_hit_rate": 0.75
  },
  "resources": {
    "cpu_usage_percent": 45,
    "memory_usage_mb": 2048,
    "gpu_usage_percent": 60
  }
}
```

## Error Responses

All errors follow a consistent format:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "request_id": "uuid",
    "details": {}
  }
}
```

### Common Error Codes

- `VALIDATION_ERROR`: Input validation failed (400)
- `AUTHENTICATION_ERROR`: Authentication failed (401)
- `AUTHORIZATION_ERROR`: Insufficient permissions (403)
- `RATE_LIMIT_ERROR`: Rate limit exceeded (429)
- `INTERNAL_SERVER_ERROR`: Unexpected server error (500)
- `SERVICE_UNAVAILABLE`: Service temporarily unavailable (503)

## Rate Limiting

The API implements tiered rate limiting:

| Role        | Requests/Minute | Requests/Hour |
|-------------|-----------------|---------------|
| Admin       | 1000            | 10000         |
| Clinician   | 100             | 1000          |
| Researcher  | 50              | 500           |
| Default     | 20              | 200           |

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1623984000
```

## Data Models

### PatientData

```typescript
{
  patient_id: string;      // Unique patient identifier
  age: number;             // Age in years
  mmse: number;            // MMSE score (0-30)
  gender?: "M" | "F" | "OTHER";
  education?: number;      // Years of education
}
```

### DiagnosisRequest

```typescript
{
  patient_data: PatientData;
  image_base64?: string;   // Base64-encoded image
  longitudinal_id?: string; // Longitudinal tracking ID
}
```

### DiagnosisResponse

```typescript
{
  patient_id: string;
  timestamp: string;
  vision_prediction: VisionResult;
  biomarker_analysis: BiomarkerResult;
  temporal_analysis: TemporalResult;
  rag_context: string[];
  explainability: ExplainabilityResult;
  ethics_audit: EthicsResult;
  final_diagnosis: FinalDiagnosis;
  confidence: number;
  approved: boolean;
}
```

## Interactive API Documentation

Interactive API documentation is available via Swagger UI and ReDoc:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## SDK Examples

### Python

```python
import requests

# Authenticate
auth_response = requests.post(
    "http://localhost:8000/api/auth/token",
    data={"username": "clinician", "password": "password123"}
)
token = auth_response.json()["access_token"]

# Make diagnosis
headers = {"Authorization": f"Bearer {token}"}
response = requests.post(
    "http://localhost:8000/diagnose",
    json={
        "patient_data": {
            "patient_id": "OAS2_0001",
            "age": 75.5,
            "mmse": 24.0
        }
    },
    headers=headers
)
diagnosis = response.json()
```

### JavaScript

```javascript
// Authenticate
const authResponse = await fetch('http://localhost:8000/api/auth/token', {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: 'username=clinician&password=password123'
});
const {access_token} = await authResponse.json();

// Make diagnosis
const response = await fetch('http://localhost:8000/diagnose', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${access_token}`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        patient_data: {
            patient_id: 'OAS2_0001',
            age: 75.5,
            mmse: 24.0
        }
    })
});
const diagnosis = await response.json();
```

## Best Practices

1. **Always use HTTPS** in production
2. **Handle rate limits** gracefully with exponential backoff
3. **Cache responses** when appropriate
4. **Validate input** before sending requests
5. **Use appropriate authentication** for protected endpoints
6. **Monitor request IDs** for debugging
7. **Implement retry logic** for transient failures

## Support

For API issues:
- Check error codes and messages
- Review request IDs in logs
- Consult troubleshooting documentation
- Contact support with request ID and error details