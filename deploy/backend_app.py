"""
DebugLLM - FastAPI Backend Service
====================================
Handles request processing, prompt formatting, authentication,
Redis caching, rate limiting, and Langfuse observability.

This service sits between the API Gateway (Nginx) and the vLLM
inference server. It manages all application-layer concerns so the
vLLM server can focus purely on fast inference.
"""

import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import redis.asyncio as aioredis
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("debugllm.backend")


# ---------------------------------------------------------------------------
# Load Configuration
# ---------------------------------------------------------------------------

def load_config(path: str = "config/model_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8001")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
API_SECRET_KEY = os.environ.get("API_SECRET_KEY", "")
CACHE_TTL = CONFIG["cache"]["redis"]["ttl_seconds"]
MAX_INPUT_CHARS = CONFIG["security"]["input_max_chars"]


# ---------------------------------------------------------------------------
# Langfuse Observability (optional — gracefully degrades if not configured)
# ---------------------------------------------------------------------------

try:
    from langfuse import Langfuse

    langfuse = Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    LANGFUSE_ENABLED = bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))
    logger.info("Langfuse observability: %s", "ENABLED" if LANGFUSE_ENABLED else "DISABLED (no keys)")
except ImportError:
    langfuse = None
    LANGFUSE_ENABLED = False
    logger.warning("Langfuse not installed — observability disabled")


# ---------------------------------------------------------------------------
# Application Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

redis_client: Optional[aioredis.Redis] = None
http_client: Optional[httpx.AsyncClient] = None
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage connection pools across the application lifecycle."""
    global redis_client, http_client

    logger.info("Starting DebugLLM backend service...")

    # Initialise Redis connection pool
    try:
        redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connection established: %s", REDIS_URL)
    except Exception as e:
        logger.warning("Redis unavailable — caching disabled: %s", e)
        redis_client = None

    # Initialise async HTTP client for vLLM calls
    http_client = httpx.AsyncClient(
        base_url=VLLM_BASE_URL,
        timeout=httpx.Timeout(60.0, connect=5.0),
    )
    logger.info("HTTP client ready — vLLM endpoint: %s", VLLM_BASE_URL)

    yield

    # Cleanup
    if http_client:
        await http_client.aclose()
    if redis_client:
        await redis_client.aclose()
    logger.info("DebugLLM backend shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="DebugLLM API",
    description=(
        "Production API for AI-powered Python bug detection and repair. "
        "Submit buggy Python code and receive a corrected version with explanation."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

security = HTTPBearer()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Validate Bearer token against the configured API secret key."""
    if not API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfiguration: API_SECRET_KEY not set",
        )
    if credentials.credentials != API_SECRET_KEY:
        logger.warning("Rejected request with invalid API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# ---------------------------------------------------------------------------
# Input Validation & Sanitisation
# ---------------------------------------------------------------------------

BLOCKED_PATTERNS = CONFIG["security"]["blocked_patterns"]


def validate_code_input(code: str) -> str:
    """
    Reject inputs exceeding size limits or containing blocked patterns.
    Blocked patterns target prompt injection and dangerous code execution.
    """
    if len(code) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input exceeds maximum length of {MAX_INPUT_CHARS} characters",
        )

    code_lower = code.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern.lower() in code_lower:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input contains a blocked pattern: '{pattern}'",
            )

    return code.strip()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class BugFixRequest(BaseModel):
    code: str = Field(..., description="Buggy Python code to fix", example="for i in range(5)\n    print(i)")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="Sampling temperature")
    max_tokens: int = Field(512, ge=1, le=1024, description="Max tokens in output")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracing")

    @validator("code")
    def code_not_empty(cls, v):
        if not v.strip():
            raise ValueError("code field must not be empty")
        return v


class BugFixResponse(BaseModel):
    fixed_code: str
    explanation: str
    model: str
    latency_ms: float
    tokens_used: int
    cache_hit: bool = False


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    redis_connected: bool
    vllm_reachable: bool


# ---------------------------------------------------------------------------
# Prompt Helpers
# ---------------------------------------------------------------------------

def build_prompt(code: str) -> str:
    return (
        "### Instruction:\n"
        "Fix the bug in the following Python code.\n\n"
        f"### Input:\n```python\n{code}\n```\n\n"
        "### Response:\n"
    )


def cache_key(code: str, temperature: float, max_tokens: int) -> str:
    payload = f"{code}|{temperature}|{max_tokens}"
    return "debugllm:cache:" + hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/fix-python-bug",
    response_model=BugFixResponse,
    summary="Fix a Python bug",
    description=(
        "Submit buggy Python code. Returns the corrected code and a brief "
        "explanation of what was wrong and how it was fixed."
    ),
)
@limiter.limit("100/minute")
async def fix_python_bug(
    request: Request,
    body: BugFixRequest,
    _: str = Depends(verify_api_key),
):
    # 1. Validate & sanitise input
    safe_code = validate_code_input(body.code)

    # 2. Check Redis cache
    ck = cache_key(safe_code, body.temperature, body.max_tokens)
    cached = None
    if redis_client:
        try:
            cached = await redis_client.get(ck)
        except Exception as e:
            logger.warning("Redis GET failed: %s", e)

    if cached:
        import json
        result = json.loads(cached)
        result["cache_hit"] = True
        logger.info("Cache HIT for key %s...", ck[:16])
        return result

    # 3. Build prompt and call vLLM
    prompt = build_prompt(safe_code)
    t0 = time.time()

    # Start Langfuse trace
    trace = None
    if LANGFUSE_ENABLED and langfuse:
        trace = langfuse.trace(
            name="fix-python-bug",
            session_id=body.session_id,
            input={"code": safe_code},
        )

    try:
        response = await http_client.post(
            "/v1/completions",
            json={
                "model": "debugllm",           # matches --lora-modules name in vLLM
                "prompt": prompt,
                "temperature": body.temperature,
                "max_tokens": body.max_tokens,
                "stop": ["### Instruction:", "<|endoftext|>"],
                "top_p": 0.95,
            },
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error("vLLM returned HTTP %s: %s", e.response.status_code, e.response.text)
        raise HTTPException(status_code=502, detail="Inference server error")
    except httpx.RequestError as e:
        logger.error("vLLM request failed: %s", e)
        raise HTTPException(status_code=503, detail="Inference server unreachable")

    latency_ms = (time.time() - t0) * 1000
    vllm_data = response.json()
    raw_text = vllm_data["choices"][0]["text"].strip()
    tokens_used = vllm_data["usage"]["completion_tokens"]

    # 4. Parse response
    lines = raw_text.split("\n")
    # Heuristic: first block is code, remainder is explanation
    split_at = next((i for i, l in enumerate(lines) if l.startswith("Explanation") or l.startswith("#")), len(lines) // 2)
    fixed_code = "\n".join(lines[:split_at]).strip().strip("```python").strip("```").strip()
    explanation = "\n".join(lines[split_at:]).strip()

    result = BugFixResponse(
        fixed_code=fixed_code,
        explanation=explanation,
        model=f"Phi-3-mini + LoRA (DebugLLM)",
        latency_ms=round(latency_ms, 2),
        tokens_used=tokens_used,
        cache_hit=False,
    ).model_dump()

    # 5. Store in cache
    if redis_client:
        try:
            import json
            await redis_client.setex(ck, CACHE_TTL, json.dumps(result))
        except Exception as e:
            logger.warning("Redis SET failed: %s", e)

    # 6. Update Langfuse trace
    if trace and LANGFUSE_ENABLED:
        trace.update(
            output={"fixed_code": fixed_code, "explanation": explanation},
            metadata={"latency_ms": latency_ms, "tokens_used": tokens_used},
        )

    logger.info(
        "Inference complete | latency=%.1fms | tokens=%d | cache=MISS",
        latency_ms,
        tokens_used,
    )
    return result


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health_check():
    """Returns service health, uptime, and dependency status."""
    redis_ok = False
    if redis_client:
        try:
            await redis_client.ping()
            redis_ok = True
        except Exception:
            pass

    vllm_ok = False
    if http_client:
        try:
            r = await http_client.get("/health", timeout=3.0)
            vllm_ok = r.status_code == 200
        except Exception:
            pass

    return HealthResponse(
        status="healthy" if vllm_ok else "degraded",
        version="1.0.0",
        uptime_seconds=round(time.time() - START_TIME, 1),
        redis_connected=redis_ok,
        vllm_reachable=vllm_ok,
    )


# ---------------------------------------------------------------------------
# Global Exception Handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again."},
    )
