"""
DebugLLM - Modal Deployment Script
====================================
Deploys the Phi-3 Mini LoRA bug-fixer model as a serverless GPU endpoint
on Modal Labs infrastructure.

Usage:
    modal deploy deploy/modal_deploy.py          # Deploy to production
    modal run deploy/modal_deploy.py::test_endpoint  # Smoke test

Requirements:
    pip install modal
    modal token new
"""

import os
import time
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# Modal App & Image Definition
# ---------------------------------------------------------------------------

APP_NAME = "debugllm-phi3-bugfixer"

# Base image with CUDA 12.1 + all inference dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.4.2",
        "transformers==4.41.2",
        "peft==0.11.1",
        "huggingface_hub==0.23.2",
        "fastapi==0.111.0",
        "uvicorn==0.30.1",
        "pydantic==2.7.3",
        "langfuse==2.25.0",
        "litellm==1.40.0",
        "python-jose[cryptography]==3.3.0",
        "slowapi==0.1.9",
        "PyYAML==6.0.1",
        "redis==5.0.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(APP_NAME, image=image)

# Persistent volume for model weights (avoids re-downloading on cold start)
model_volume = modal.Volume.from_name("debugllm-model-weights", create_if_missing=True)

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
# Set these with: modal secret create debugllm-secrets HF_TOKEN=... LANGFUSE_SECRET_KEY=... etc.
secrets = modal.Secret.from_name("debugllm-secrets")

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
LORA_ADAPTER = "Sud1212/phi3-debug-llm-lora"
MODEL_DIR = "/vol/models"
MAX_OUTPUT_TOKENS = 512
GPU_MEMORY_UTILIZATION = 0.85

SYSTEM_PROMPT = """You are DebugLLM, an expert Python debugging assistant. 
When given buggy Python code, you identify the bug and return ONLY the corrected code 
followed by a brief explanation of the fix. Do not add unnecessary imports or change 
logic beyond fixing the reported bug."""

# ---------------------------------------------------------------------------
# Prompt Formatting
# ---------------------------------------------------------------------------

def build_prompt(buggy_code: str) -> str:
    """Format the buggy code into the model's instruction template."""
    return (
        f"### Instruction:\n"
        f"Fix the bug in the following Python code.\n\n"
        f"### Input:\n"
        f"```python\n{buggy_code.strip()}\n```\n\n"
        f"### Response:\n"
    )


def parse_response(raw: str) -> dict:
    """Split model output into fixed_code and explanation."""
    lines = raw.strip().split("\n")
    code_lines, explanation_lines = [], []
    in_code_block = False
    found_explanation = False

    for line in lines:
        if line.startswith("```python"):
            in_code_block = True
            continue
        elif line.startswith("```") and in_code_block:
            in_code_block = False
            found_explanation = True
            continue

        if in_code_block:
            code_lines.append(line)
        elif found_explanation or not line.startswith("```"):
            explanation_lines.append(line)

    # Fallback: if no code block markers, treat first block as code
    if not code_lines:
        code_lines = lines[:max(len(lines) // 2, 1)]
        explanation_lines = lines[len(code_lines):]

    return {
        "fixed_code": "\n".join(code_lines).strip(),
        "explanation": "\n".join(explanation_lines).strip(),
    }


# ---------------------------------------------------------------------------
# Input / Output Schemas
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field, validator


class BugFixRequest(BaseModel):
    code: str = Field(..., description="The buggy Python code to fix", max_length=8000)
    temperature: float = Field(0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(512, ge=1, le=1024)
    session_id: Optional[str] = Field(None, description="Optional session ID for tracing")

    @validator("code")
    def code_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("code must not be empty")
        return v


class BugFixResponse(BaseModel):
    fixed_code: str
    explanation: str
    model: str
    latency_ms: float
    tokens_used: int


class HealthResponse(BaseModel):
    status: str
    model: str
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Modal Class — GPU Inference Server
# ---------------------------------------------------------------------------

@app.cls(
    gpu=modal.gpu.A10G(),
    secrets=[secrets],
    volumes={MODEL_DIR: model_volume},
    container_idle_timeout=300,     # Keep warm for 5 min after last request
    allow_concurrent_inputs=32,
    timeout=120,
)
class DebugLLMService:
    """
    Serverless GPU inference service for the DebugLLM bug-fixing model.

    Lifecycle:
        - __enter__: Downloads model weights (cached in Volume), loads vLLM engine
        - fix_bug: Handles individual inference requests
        - health: Returns service health status
        - __exit__: Cleanup
    """

    @modal.enter()
    def load_model(self):
        """Load the model once at container startup."""
        import torch
        from vllm import LLM
        from huggingface_hub import snapshot_download

        self._start_time = time.time()
        print(f"[DebugLLM] Loading model: {MODEL_ID} + LoRA: {LORA_ADAPTER}")

        # Download weights to persistent volume (skipped on warm starts)
        base_path = os.path.join(MODEL_DIR, "phi3-base")
        lora_path = os.path.join(MODEL_DIR, "phi3-lora")

        if not os.path.exists(base_path):
            print("[DebugLLM] Downloading base model weights...")
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=base_path,
                token=os.environ.get("HF_TOKEN"),
            )
            model_volume.commit()

        if not os.path.exists(lora_path):
            print("[DebugLLM] Downloading LoRA adapter weights...")
            snapshot_download(
                repo_id=LORA_ADAPTER,
                local_dir=lora_path,
                token=os.environ.get("HF_TOKEN"),
            )
            model_volume.commit()

        # Initialize vLLM engine
        self.llm = LLM(
            model=base_path,
            enable_lora=True,
            max_lora_rank=16,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_num_seqs=32,
            dtype="float16",
        )
        self.lora_path = lora_path
        print("[DebugLLM] Model loaded and ready.")

    @modal.method()
    def fix_bug(self, request_data: dict) -> dict:
        """
        Run inference on a single bug-fix request.

        Args:
            request_data: Serialized BugFixRequest dict

        Returns:
            Serialized BugFixResponse dict
        """
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        req = BugFixRequest(**request_data)
        t0 = time.time()

        prompt = build_prompt(req.code)
        sampling_params = SamplingParams(
            temperature=req.temperature,
            top_p=0.95,
            max_tokens=req.max_tokens,
            stop=["### Instruction:", "<|endoftext|>"],
        )

        outputs = self.llm.generate(
            [prompt],
            sampling_params,
            lora_request=LoRARequest("debugllm-lora", 1, self.lora_path),
        )

        raw_output = outputs[0].outputs[0].text
        tokens_used = len(outputs[0].outputs[0].token_ids)
        parsed = parse_response(raw_output)
        latency_ms = (time.time() - t0) * 1000

        return BugFixResponse(
            fixed_code=parsed["fixed_code"],
            explanation=parsed["explanation"],
            model=f"{MODEL_ID}+LoRA",
            latency_ms=round(latency_ms, 2),
            tokens_used=tokens_used,
        ).model_dump()

    @modal.method()
    def health(self) -> dict:
        return HealthResponse(
            status="healthy",
            model=MODEL_ID,
            uptime_seconds=round(time.time() - self._start_time, 1),
        ).model_dump()


# ---------------------------------------------------------------------------
# FastAPI Web Endpoint (wraps the Modal class)
# ---------------------------------------------------------------------------

@app.function(
    secrets=[secrets],
    allow_concurrent_inputs=50,
)
@modal.asgi_app()
def fastapi_app():
    """ASGI wrapper exposing the DebugLLM service as a REST API."""
    from fastapi import FastAPI, HTTPException, Depends, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

    web_app = FastAPI(
        title="DebugLLM API",
        description="Production API for AI-powered Python bug fixing",
        version="1.0.0",
    )

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    security = HTTPBearer()
    service = DebugLLMService()

    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
        expected = os.environ.get("API_SECRET_KEY", "")
        if credentials.credentials != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )
        return credentials.credentials

    @web_app.post("/fix-python-bug", response_model=BugFixResponse)
    async def fix_python_bug(
        request: BugFixRequest,
        _: str = Depends(verify_token),
    ):
        """
        Accepts buggy Python code and returns the fixed version with explanation.

        - **code**: The Python code containing a bug (required)
        - **temperature**: Sampling temperature — lower = more deterministic (default: 0.1)
        - **max_tokens**: Max tokens in the response (default: 512)
        """
        try:
            result = service.fix_bug.remote(request.model_dump())
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    @web_app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Returns service health and uptime."""
        return service.health.remote()

    return web_app


# ---------------------------------------------------------------------------
# Local Test Function
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def test_endpoint():
    """Quick smoke test — run with: modal run deploy/modal_deploy.py"""
    service = DebugLLMService()

    test_cases = [
        {
            "code": "for i in range(5)\n    print(i)",
            "description": "Missing colon in for loop",
        },
        {
            "code": "numbers = [1,2,3,4]\nprint(numbers[4])",
            "description": "Index out of range",
        },
        {
            "code": "def greet(name):\nprint('Hello, ' + name)",
            "description": "Missing indentation in function body",
        },
    ]

    print("\n" + "=" * 60)
    print("DebugLLM Smoke Test")
    print("=" * 60)

    for i, tc in enumerate(test_cases, 1):
        print(f"\nTest {i}: {tc['description']}")
        print(f"Input:\n{tc['code']}\n")

        result = service.fix_bug.remote({
            "code": tc["code"],
            "temperature": 0.1,
            "max_tokens": 256,
        })

        print(f"Fixed Code:\n{result['fixed_code']}")
        print(f"Explanation: {result['explanation']}")
        print(f"Latency: {result['latency_ms']}ms | Tokens: {result['tokens_used']}")
        print("-" * 40)

    print("\n✅ All smoke tests completed.")
