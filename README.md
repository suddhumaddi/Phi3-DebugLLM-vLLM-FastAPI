# DebugLLM — Production Deployment for AI-Powered Python Bug Fixing

> **A production-ready deployment architecture for Phi-3 Mini + LoRA, serving an automated Python bug-fixing assistant via a secure, scalable, observable REST API.**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![vLLM](https://img.shields.io/badge/Inference-vLLM%200.4.2-orange.svg)](https://github.com/vllm-project/vllm)
[![Modal](https://img.shields.io/badge/Deploy-Modal%20Labs-purple.svg)](https://modal.com)

---

## Overview

This repository contains the **Module 2 Capstone** for the ReadyTensor LLM Engineering & Deployment (LLMED) Certification. It provides a complete, production-grade deployment architecture for the **DebugLLM** system — a fine-tuned Phi-3 Mini model (from Module 1) that automatically detects and repairs common Python code bugs.

The deployment covers:
- A **serverless GPU endpoint** on Modal Labs (primary) and a **self-hosted Docker Compose** stack (alternative)
- A **FastAPI backend** with authentication, input validation, Redis caching, and rate limiting
- A **vLLM inference engine** for high-throughput, low-latency model serving
- A **monitoring and observability stack** using Langfuse, LiteLLM, and CloudWatch
- A **Python client** with full error handling and a test suite

This is an **engineering design project** — it demonstrates production LLM deployment knowledge. Actual cloud deployment requires valid API keys and cloud accounts (see [Environment Setup](#environment-setup)).

---

## Target Audience

This project is aimed at:
- ML engineers learning LLM deployment and production engineering
- Students completing the ReadyTensor LLMED certification
- Developers who want a reference architecture for deploying fine-tuned LLMs

**Prerequisites:** Familiarity with Python, REST APIs, Docker, and basic cloud infrastructure concepts.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Users                                │
│  (Web UI / IDE Plugin / Developer API)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTPS
┌──────────────────────▼──────────────────────────────────────┐
│               Nginx / API Gateway                           │
│   TLS Termination · Rate Limiting · Request Filtering       │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────────────┐
│              FastAPI Backend  (Port 8000)                   │
│  Auth · Input Validation · Prompt Formatting · Caching      │
└────────┬────────────────────────────────────┬───────────────┘
         │ HTTP/OpenAI API                    │
┌────────▼──────────┐             ┌───────────▼───────────────┐
│  vLLM Inference   │             │      Redis Cache           │
│  Server (8001)    │             │  (Identical prompt cache)  │
│  Phi-3 + LoRA     │             └───────────────────────────┘
└────────┬──────────┘
         │
┌────────▼──────────────────────────────────────────────────┐
│        GPU Instance (NVIDIA A10G, 24GB VRAM)              │
└───────────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────────┐
│              Monitoring Stack                             │
│  Langfuse (LLM traces) · CloudWatch · LiteLLM (cost)     │
└───────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
debugllm-deployment/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── .gitignore
├── .env.example                 # Environment variable template
├── requirements.txt             # Python dependencies (pinned versions)
├── Dockerfile                   # Multi-stage production image for backend
│
├── deploy/
│   ├── modal_deploy.py          # Serverless GPU deployment on Modal Labs
│   ├── docker-compose.yml       # Self-hosted stack (vLLM + FastAPI + Redis + Nginx)
│   └── backend_app.py           # FastAPI application (auth, caching, inference proxy)
│
├── client/
│   ├── client.py                # Python client with error handling & retry logic
│   └── test_requests.py         # Integration test suite for live endpoint
│
└── config/
    └── model_config.yaml        # Model, inference, infrastructure, and security config
```

---

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/suddhumaddi/debugllm-deployment.git
cd debugllm-deployment
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
# Open .env in your editor and fill in all required values
```

Required variables:

| Variable | Description | Required |
|---|---|---|
| `HF_TOKEN` | HuggingFace access token (for model download) | ✅ |
| `API_SECRET_KEY` | Bearer token for API authentication | ✅ |
| `REDIS_PASSWORD` | Redis authentication password | ✅ (Docker) |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key for observability | Optional |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | Optional |

---

## Deployment Options

### Option A — Modal Labs (Recommended)

Modal provides serverless GPU compute — you only pay when requests are being processed.

**Step 1: Install Modal and authenticate**

```bash
pip install modal
modal token new        # Opens browser for authentication
```

**Step 2: Set secrets**

```bash
modal secret create debugllm-secrets \
    HF_TOKEN="hf_..." \
    API_SECRET_KEY="your-secret-key" \
    LANGFUSE_PUBLIC_KEY="pk-lf-..." \
    LANGFUSE_SECRET_KEY="sk-lf-..."
```

**Step 3: Deploy**

```bash
modal deploy deploy/modal_deploy.py
```

Modal will output a URL like: `https://your-org--debugllm-phi3-bugfixer-fastapi-app.modal.run`

**Step 4: Smoke test**

```bash
modal run deploy/modal_deploy.py
```

---

### Option B — Self-Hosted Docker Compose

Suitable for on-premises deployments or dedicated GPU servers.

**Requirements:**
- Docker >= 24.0
- NVIDIA Container Toolkit installed
- GPU with >= 16GB VRAM (e.g., RTX 3090, A10G, A100)

**Step 1: Set environment variables**

```bash
cp .env.example .env
# Fill in HF_TOKEN, API_SECRET_KEY, REDIS_PASSWORD
```

**Step 2: Start the stack**

```bash
docker compose -f deploy/docker-compose.yml up -d
```

This starts four containers:
- `debugllm-vllm` — vLLM inference server on port 8001
- `debugllm-backend` — FastAPI backend on port 8000
- `debugllm-redis` — Redis cache
- `debugllm-nginx` — Nginx reverse proxy on ports 80/443

**Step 3: Wait for model loading (~2 minutes)**

```bash
docker compose logs -f vllm
# Wait until you see: "Application startup complete"
```

**Step 4: Verify**

```bash
curl http://localhost:8000/health
```

---

## Using the Client

### Quick Start

```python
from client.client import DebugLLMClient

client = DebugLLMClient(
    base_url="https://your-endpoint.modal.run",  # or http://localhost:8000
    api_key="your-api-secret-key"
)

result = client.fix_bug("for i in range(5)\n    print(i)")
print(result.fixed_code)
# Output: for i in range(5):
#             print(i)
print(result.explanation)
# Output: The for loop was missing the required colon ':' ...
```

### Batch Processing

```python
snippets = [
    "for i in range(5)\n    print(i)",
    "x = [1,2,3]\nprint(x[3])",
    "def greet(name):\nprint('Hello ' + name)"
]

results = client.fix_bugs_batch(snippets)
for r in results:
    print(r)
```

### API Directly (curl)

```bash
curl -X POST https://your-endpoint/fix-python-bug \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"code": "for i in range(5)\n    print(i)"}'
```

**Response:**

```json
{
  "fixed_code": "for i in range(5):\n    print(i)",
  "explanation": "The for loop statement was missing the required colon ':' that marks the beginning of the loop block in Python.",
  "model": "Phi-3-mini + LoRA (DebugLLM)",
  "latency_ms": 820.4,
  "tokens_used": 48,
  "cache_hit": false
}
```

---

## Running Tests

```bash
# Set your endpoint and key
export DEBUGLLM_BASE_URL="http://localhost:8000"
export DEBUGLLM_API_KEY="your-api-secret-key"

# Run the integration test suite
python client/test_requests.py
```

The test suite covers:
- Health check endpoint
- Happy-path inference for 4 bug types
- Response schema validation
- Redis cache behaviour (cache MISS on first call, HIT on second)
- Authentication rejection with invalid key
- Input validation (empty input, oversized input, blocked patterns)
- Temperature parameter variants
- Batch processing

---

## Configuration Reference

All deployment settings are managed in `config/model_config.yaml`. Key sections:

| Section | Key Settings |
|---|---|
| `model` | Model ID, quantization, context length, max output tokens |
| `inference` | vLLM engine settings, batching, prefix caching |
| `generation` | Temperature, top_p, stop sequences |
| `serving` | Host, port, rate limits |
| `infrastructure` | Cloud provider, instance type, auto-scaling thresholds |
| `monitoring` | Langfuse, CloudWatch, LiteLLM settings |
| `cache` | Redis TTL, memory limits |
| `security` | JWT expiry, blocked patterns, input size limits |

---

## API Reference

### `POST /fix-python-bug`

Submits buggy Python code for automated repair.

**Authentication:** `Authorization: Bearer <API_SECRET_KEY>`

**Request Body:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `code` | string | ✅ | — | Buggy Python code (max 8,000 chars) |
| `temperature` | float | ❌ | 0.1 | Sampling temperature (0.0–1.0) |
| `max_tokens` | int | ❌ | 512 | Max tokens in output (1–1024) |
| `session_id` | string | ❌ | null | Session ID for Langfuse tracing |

**Response:**

| Field | Type | Description |
|---|---|---|
| `fixed_code` | string | The corrected Python code |
| `explanation` | string | Plain-language description of the fix |
| `model` | string | Model identifier used |
| `latency_ms` | float | End-to-end inference latency |
| `tokens_used` | int | Number of output tokens generated |
| `cache_hit` | bool | Whether response was served from cache |

**Error Codes:**

| Code | Meaning |
|---|---|
| 400 | Invalid input (empty, too long, or blocked pattern) |
| 401 | Missing or invalid API key |
| 422 | Request body validation error |
| 429 | Rate limit exceeded (100 req/min per user) |
| 502 | vLLM inference server error |
| 503 | vLLM server unreachable |

### `GET /health`

Returns service health and dependency status. No authentication required.

---

## Monitoring & Observability

### Langfuse (LLM Tracing)

Every inference request is traced in Langfuse with:
- Input prompt and output
- Token counts and cost estimates
- Latency per request
- Session grouping for multi-turn analysis

Dashboard: [https://cloud.langfuse.com](https://cloud.langfuse.com)

### CloudWatch / Datadog (Infrastructure)

Key metrics captured:
- GPU utilisation and VRAM usage
- Request throughput (req/sec)
- Error rate by status code
- Container CPU and memory

### LiteLLM (Cost Tracking)

Tracks cumulative token spend per request and per day, with configurable budget alerts.

---

## Troubleshooting

**vLLM container doesn't start:**
```bash
docker compose logs vllm
# Common cause: insufficient VRAM — check GPU with: nvidia-smi
```

**401 Unauthorized:**
```bash
# Verify your API key matches what's in .env
echo $DEBUGLLM_API_KEY
```

**Model not found / HF download failure:**
```bash
# Verify HF_TOKEN is set and has read access
huggingface-cli whoami
```

**Redis connection refused:**
```bash
docker compose ps redis       # check it's running
docker compose logs redis     # check for auth errors
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

Model weights are subject to Microsoft's Phi-3 license. Please review the [Phi-3 model card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) before commercial use.

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change, then submit a pull request against the `main` branch.

For bug reports, please include your Python version, deployment method, and the full error traceback.

---

## Contact

**Sudarshan Maddi**  
Woxsen University  
🔗 [GitHub](https://github.com/suddhumaddi) | 🤗 [HuggingFace](https://huggingface.co/Sud1212)

For questions or issues, please open a [GitHub Issue](https://github.com/suddhumaddi/debugllm-deployment/issues).

---

## Related Resources

| Resource | Link |
|---|---|
| Module 1 — Fine-Tuning Publication | [ReadyTensor Publication](https://readytensor.ai) |
| Fine-Tuned Model | [Sud1212/phi3-debug-llm-lora](https://huggingface.co/Sud1212/phi3-debug-llm-lora) |
| Base Model | [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |
| Training Dataset | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) |
| W&B Training Dashboard | [wandb.ai](https://wandb.ai/suddhumaddi-woxsen-university/huggingface) |
