# OASIS Agentic Pipeline Architecture

Comprehensive architecture documentation for the OASIS Agentic Pipeline multi-agent AI system.

## System Overview

The OASIS Agentic Pipeline is a sophisticated multi-agent AI system for Alzheimer's disease diagnosis that combines computer vision, clinical biomarker analysis, temporal tracking, knowledge retrieval, and ethical reasoning to provide comprehensive diagnostic insights.

### Key Characteristics

- **Multi-Agent Architecture**: 12 specialized AI agents working collaboratively
- **Hybrid Edge-Cloud Deployment**: Local Ollama + tiered Claude (Haiku→Sonnet→Opus)
- **Snapdragon NPU Optimization**: Windows ARM64 with hardware acceleration
- **Ethical AI Integration**: Built-in ethical guardrails and bias detection
- **Real-Time Inference**: Sub-second diagnosis with explainability
- **Scalable Design**: Batch processing and async task support

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     API Layer (FastAPI)                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ REST API │ │ WebSocket│ │  GraphQL │ │  gRPC    │          │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘          │
└───────┼────────────┼────────────┼────────────┼─────────────────┘
        │            │            │            │
┌───────┴────────────┴────────────┴────────────┴─────────────────┐
│                   Middleware Layer                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Security │ │ Rate     │ │ Caching  │ │  Async   │          │
│  │          │ │ Limiting │ │          │ │  Tasks   │          │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘          │
└───────┼────────────┼────────────┼────────────┼─────────────────┘
        │            │            │            │
┌───────┴────────────┴────────────┴────────────┴─────────────────┐
│                 Chief Medical Officer (CMO)                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Agent Orchestration & Coordination          │  │
│  └────────────────────┬─────────────────────────────────────┘  │
└───────────────────────┼─────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────┴─────┐ ┌──────┴──────┐ ┌─────┴────────┐
│   Vision    │ │  Biomarker  │ │   Temporal   │
│   Agent     │ │   Agent     │ │   Analyst    │
└─────────────┘ └─────────────┘ └──────────────┘
┌─────────────┐ ┌─────────────┐ ┌──────────────┐
│   RAG       │ │  Ethicist   │ │  Reasoner    │
│   Agent     │ │   Agent     │ │   Agent      │
└─────────────┘ └─────────────┘ └──────────────┘
┌─────────────┐ ┌─────────────┐ ┌──────────────┐
│ Volumetry   │ │ ATN         │ │ Differential  │
│   Agent     │ │ Profiler    │ │   Agent      │
└─────────────┘ └─────────────┘ └──────────────┘
┌─────────────┐ ┌─────────────┐
│ Therapeutic │ │ Explainer   │
│   Agent     │ │   Agent     │
└─────────────┘ └─────────────┘
                        │
┌───────────────────────┴─────────────────────────────────────────┐
│                    Data & Model Layer                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ OASIS    │ │ Clinical │ │ Vector   │ │  Model   │          │
│  │ Dataset  │ │  Data    │ │  Store   │ │  Cache   │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
└─────────────────────────────────────────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────────────────┐
│                  Infrastructure Layer                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │   GPU/   │ │   CPU/   │ │   Redis  │ │ Monitor  │          │
│  │   NPU    │ │   Edge   │ │  Cache   │ │  & Logs  │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Architecture

### 1. Vision Agent
**Purpose**: MRI image analysis and classification
- **Model**: CNN-based (ResNet/EfficientNet backbone)
- **Input**: 224x224 grayscale MRI images
- **Output**: 4-class probability distribution
- **Optimization**: GPU acceleration, quantization support

### 2. Biomarker Agent
**Purpose**: Clinical biomarker analysis and interpretation
- **Input**: MMSE, age, education, SES, eTIV, nWBV, ASF
- **Processing**: Normalization, age adjustment, risk factor analysis
- **Output**: Biomarker interpretation and risk assessment

### 3. Temporal Analyst Agent
**Purpose**: Longitudinal tracking and progression analysis
- **Input**: Multi-visit clinical data
- **Processing**: Trajectory calculation, atrophy velocity
- **Output**: Progression trends and velocity metrics

### 4. RAG Agent (Medical Librarian)
**Purpose**: Knowledge retrieval from medical literature
- **Input**: Clinical queries and context
- **Processing**: Vector similarity search, relevance ranking
- **Output**: Relevant medical guidelines and research findings

### 5. Ethicist Agent
**Purpose**: Ethical guardrails and bias detection
- **Input**: Diagnostic proposals and confidence scores
- **Processing**: Consistency checking, bias detection, confidence validation
- **Output**: Ethical approval/rejection with reasoning

### 6. Clinical Reasoner Agent
**Purpose**: Evidence synthesis and clinical reasoning
- **Input**: Multi-agent evidence and context
- **Processing**: Evidence weighting, LLM-powered synthesis
- **Output**: Clinical narrative with tier assignment

### 7. Volumetry Agent
**Purpose**: Regional brain volume analysis
- **Input**: FreeSurfer data or biomarker estimates
- **Processing**: MTA score calculation, regional analysis
- **Output**: Volumetry summary and risk assessment

### 8. ATN Profiler Agent
**Purpose**: Amyloid-Tau-Neurodegeneration classification
- **Input**: PET biomarkers (amyloid, tau), MRI measures
- **Processing**: ATN framework classification
- **Output**: ATN status and category

### 9. Differential Agent
**Purpose**: Multi-etiology differential diagnosis
- **Input**: Comprehensive evidence from all agents
- **Processing**: Etiology ranking, workup recommendations
- **Output**: Differential diagnosis with likelihood scores

### 10. Therapeutic Agent
**Purpose**: Treatment recommendations and research hypotheses
- **Input**: Patient profile and diagnosis
- **Processing**: Evidence-based treatment matching
- **Output**: Therapeutic options and research hypotheses

### 11. Explainer Agent
**Purpose**: Model explainability and visualization
- **Input**: Image and prediction
- **Processing**: Grad-CAM heatmap generation
- **Output**: Attention maps and interpretability visualizations

### 12. Chief Medical Officer (CMO)
**Purpose**: Agent orchestration and final decision making
- **Input**: All agent outputs
- **Processing**: Evidence synthesis, ethical validation, tier assignment
- **Output**: Final diagnosis with confidence and explanation

## Data Flow

### Diagnosis Request Flow

```
1. API receives diagnosis request
   ↓
2. Authentication & validation
   ↓
3. Image preprocessing & validation
   ↓
4. CMO orchestrates parallel agent execution:
   ├─ Vision Agent processes image
   ├─ Biomarker Agent processes clinical data
   ├─ Temporal Agent analyzes longitudinal data
   ├─ RAG Agent retrieves medical context
   └─ Explainer Agent generates heatmap
   ↓
5. Ethicist Agent validates results
   ↓
6. Clinical Reasoner synthesizes evidence
   ↓
7. CMO makes final decision with tier assignment
   ↓
8. Response returned with full explanation
```

### LLM Provider Flow

```
Request arrives
   ↓
Check confidence threshold
   ↓
If confidence < threshold:
   ├─ Try local Ollama (free tier)
   ├─ If fails/low confidence: escalate to Claude Haiku (cheap tier)
   ├─ If fails/low confidence: escalate to Claude Sonnet (standard tier)
   └─ If fails/low confidence: escalate to Claude Opus (deep tier)
   ↓
If confidence >= threshold:
   └─ Use current tier
   ↓
Return result with tier info
```

## Component Interactions

### Agent Communication

```python
# Agent interaction pattern
class AgentInteraction:
    def __init__(self):
        self.agents = {
            'vision': VisionAgent(),
            'biomarker': BiomarkerAgent(),
            # ... other agents
        }
    
    def execute_diagnosis(self, patient_data, image):
        # Parallel execution
        with ThreadPoolExecutor() as executor:
            vision_future = executor.submit(
                self.agents['vision'].predict, image
            )
            biomarker_future = executor.submit(
                self.agents['biomarker'].analyze, patient_data
            )
            
            vision_result = vision_future.result()
            biomarker_result = biomarker_future.result()
        
        # Sequential dependent execution
        ethicist_result = self.agents['ethicist'].audit(
            vision_result, biomarker_result
        )
        
        return ethicist_result
```

### Error Handling Flow

```
Agent execution attempt
   ↓
Exception caught
   ↓
Check exception type
   ├─ Retryable error → Retry with exponential backoff
   ├─ Circuit breaker open → Fail fast
   └─ Non-retryable error → Log and escalate
   ↓
Graceful degradation
   ↓
Return partial results or error response
```

## Security Architecture

### Authentication Flow

```
Client request
   ↓
JWT token validation
   ↓
Role extraction from token
   ↓
Permission check for endpoint
   ↓
Rate limit check
   ↓
Request processing
   ↓
Response with security headers
```

### Data Protection

- **Encryption**: TLS 1.3 for all communications
- **Authentication**: OAuth2/OIDC with JWT tokens
- **Authorization**: Role-based access control (RBAC)
- **Input Validation**: Comprehensive sanitization and validation
- **Security Headers**: HSTS, CSP, X-Frame-Options, etc.
- **Audit Logging**: Request tracking with correlation IDs

## Performance Architecture

### Caching Strategy

```
Request → L1 Cache (In-Memory) → L2 Cache (Redis) → Compute
         ↓ Hit                    ↓ Hit               ↓ Miss
     Return Result          Return Result      Execute & Cache
```

### Async Processing

```
Long-running task submitted
   ↓
Added to task queue
   ↓
Worker thread/process picks up
   ↓
Execute in background
   ↓
Store result
   ↓
Client polls for result
```

## Deployment Architecture

### Development Environment

```
Developer Machine
├─ Local Python environment
├─ Ollama for local LLM
├─ SQLite for development DB
└─ In-memory caching
```

### Production Environment

```
Kubernetes Cluster
├─ API Pods (FastAPI)
├─ Worker Pods (Background tasks)
├─ Redis Cluster (Caching)
├─ PostgreSQL (Production DB)
├─ Prometheus (Monitoring)
└─ Grafana (Visualization)
```

## Scalability Considerations

### Horizontal Scaling

- **API Layer**: Stateless pods, auto-scaling based on CPU/memory
- **Worker Layer**: Independent scaling for background tasks
- **Cache Layer**: Redis cluster for distributed caching
- **Database Layer**: Read replicas for query scaling

### Vertical Scaling

- **GPU/NPU**: Model inference acceleration
- **Memory**: Larger cache sizes
- **CPU**: More parallel processing workers

## Monitoring Architecture

### Metrics Collection

```
Application → Metrics Exporter → Prometheus → Grafana
            ↓
         Alert Manager → Notifications
```

### Logging

```
Application → Structured Logs → Log Aggregation → SIEM
            ↓
         Error Tracking → Alerting
```

### Tracing

```
Request → Distributed Tracing → Trace Analysis → Performance Insights
```

## Technology Stack

### Core Technologies

- **Language**: Python 3.10+
- **Web Framework**: FastAPI
- **ML Framework**: PyTorch
- **Image Processing**: PIL, OpenCV
- **Data Processing**: Pandas, NumPy

### Infrastructure

- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus, Grafana
- **Logging**: Structured logging with JSON

### AI/ML

- **Vision Models**: ResNet, EfficientNet
- **LLM Integration**: Anthropic Claude, Ollama
- **Vector Database**: ChromaDB
- **Model Optimization**: ONNX, Quantization

## Configuration Management

### Environment-Specific Configs

```python
# Development
DEBUG=true
LOG_LEVEL=debug
LLM_PROVIDER=ollama
CACHE_PROVIDER=memory

# Production
DEBUG=false
LOG_LEVEL=info
LLM_PROVIDER=anthropic
CACHE_PROVIDER=redis
```

### Feature Flags

```python
ENABLE_NEW_AGENT=true
USE_GPU_INFERENCE=true
ENABLE_ASYNC_PROCESSING=true
```

## Disaster Recovery

### Backup Strategy

- **Database**: Daily backups, point-in-time recovery
- **Models**: Versioned in artifact registry
- **Configuration**: Git-tracked with environment overrides
- **Cache**: Redis persistence and replication

### Failover

- **API**: Multi-zone deployment with load balancing
- **Cache**: Redis cluster with automatic failover
- **Database**: Primary-replica with automatic promotion

## Future Enhancements

### Planned Improvements

1. **Model Optimization**: ONNX export and quantization
2. **Advanced Caching**: Predictive preloading
3. **Edge Deployment**: ONNX Runtime for edge devices
4. **Federated Learning**: Privacy-preserving model updates
5. **Real-Time Streaming**: WebSocket-based real-time analysis

This architecture provides a robust, scalable, and maintainable foundation for the OASIS Agentic Pipeline with clear separation of concerns and well-defined interfaces between components.