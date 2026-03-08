"""
DebugLLM - Python Client
=========================
A lightweight client for interacting with the DebugLLM API.
Supports single requests, batch processing, and basic error handling
with retry logic.

Usage:
    from client.client import DebugLLMClient

    client = DebugLLMClient(base_url="https://your-endpoint", api_key="sk-...")
    result = client.fix_bug("for i in range(5)\n    print(i)")
    print(result.fixed_code)

Or run directly:
    python client/client.py
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("debugllm.client")


# ---------------------------------------------------------------------------
# Response Dataclass
# ---------------------------------------------------------------------------

@dataclass
class BugFixResult:
    """Structured response from the DebugLLM API."""
    fixed_code: str
    explanation: str
    model: str
    latency_ms: float
    tokens_used: int
    cache_hit: bool

    def __str__(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"FIXED CODE:\n{self.fixed_code}\n"
            f"\nEXPLANATION:\n{self.explanation}\n"
            f"\nModel: {self.model} | "
            f"Latency: {self.latency_ms:.1f}ms | "
            f"Tokens: {self.tokens_used} | "
            f"Cache: {'HIT' if self.cache_hit else 'MISS'}\n"
            f"{'='*60}"
        )


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class DebugLLMError(Exception):
    """Base exception for DebugLLM client errors."""


class AuthenticationError(DebugLLMError):
    """Raised when API key is invalid or missing."""


class RateLimitError(DebugLLMError):
    """Raised when the rate limit has been exceeded."""


class InferenceError(DebugLLMError):
    """Raised when the inference server returns an error."""


class ValidationError(DebugLLMError):
    """Raised when input validation fails."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class DebugLLMClient:
    """
    Client for the DebugLLM Python Bug Fixing API.

    Args:
        base_url: Base URL of the deployed API endpoint.
                  Defaults to DEBUGLLM_BASE_URL environment variable.
        api_key:  Bearer token for authentication.
                  Defaults to DEBUGLLM_API_KEY environment variable.
        timeout:  Request timeout in seconds (default: 60).
        max_retries: Number of retry attempts on transient failures (default: 3).
    """

    DEFAULT_BASE_URL = "http://localhost:8000"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.base_url = (
            base_url
            or os.environ.get("DEBUGLLM_BASE_URL", self.DEFAULT_BASE_URL)
        ).rstrip("/")

        self.api_key = api_key or os.environ.get("DEBUGLLM_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "No API key provided. Set DEBUGLLM_API_KEY environment variable "
                "or pass api_key= to the constructor."
            )

        self.timeout = timeout

        # Configure retry strategy for transient network errors
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session = requests.Session()
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "DebugLLM-Python-Client/1.0.0",
        })

    def fix_bug(
        self,
        code: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
        session_id: Optional[str] = None,
    ) -> BugFixResult:
        """
        Submit buggy Python code for automated repair.

        Args:
            code:        The Python code containing a bug.
            temperature: Sampling temperature (0.0–1.0). Lower = more deterministic.
                         Default 0.1 is recommended for code generation.
            max_tokens:  Maximum tokens in the model response (1–1024).
            session_id:  Optional identifier for grouping traces in Langfuse.

        Returns:
            BugFixResult with fixed_code, explanation, and metadata.

        Raises:
            ValidationError:    Input is empty or exceeds length limits.
            AuthenticationError: API key is invalid.
            RateLimitError:     Rate limit exceeded — wait and retry.
            InferenceError:     Server-side error during inference.
            DebugLLMError:      Other unexpected errors.
        """
        if not code or not code.strip():
            raise ValidationError("code argument must not be empty")

        payload = {
            "code": code.strip(),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if session_id:
            payload["session_id"] = session_id

        try:
            response = self._session.post(
                f"{self.base_url}/fix-python-bug",
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError as e:
            raise DebugLLMError(f"Could not connect to {self.base_url}: {e}") from e
        except requests.exceptions.Timeout:
            raise DebugLLMError(f"Request timed out after {self.timeout}s") from None

        self._handle_http_errors(response)

        data = response.json()
        return BugFixResult(
            fixed_code=data["fixed_code"],
            explanation=data["explanation"],
            model=data["model"],
            latency_ms=data["latency_ms"],
            tokens_used=data["tokens_used"],
            cache_hit=data.get("cache_hit", False),
        )

    def fix_bugs_batch(self, code_snippets: list[str], **kwargs) -> list[BugFixResult]:
        """
        Fix multiple code snippets sequentially.

        Args:
            code_snippets: List of buggy Python code strings.
            **kwargs:      Additional keyword arguments passed to fix_bug().

        Returns:
            List of BugFixResult objects in the same order as input.
        """
        results = []
        for i, snippet in enumerate(code_snippets, 1):
            logger.info("Processing snippet %d/%d...", i, len(code_snippets))
            result = self.fix_bug(snippet, **kwargs)
            results.append(result)
        return results

    def health(self) -> dict:
        """
        Check the health of the deployed service.

        Returns:
            Dict with keys: status, version, uptime_seconds,
                            redis_connected, vllm_reachable
        """
        try:
            response = self._session.get(
                f"{self.base_url}/health",
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise DebugLLMError(f"Health check failed: {e}") from e

    def _handle_http_errors(self, response: requests.Response) -> None:
        """Map HTTP status codes to typed exceptions."""
        if response.status_code == 200:
            return

        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text

        if response.status_code == 401:
            raise AuthenticationError(f"Authentication failed: {detail}")
        elif response.status_code == 422:
            raise ValidationError(f"Input validation failed: {detail}")
        elif response.status_code == 429:
            raise RateLimitError(f"Rate limit exceeded. Retry after a moment. Detail: {detail}")
        elif response.status_code in (500, 502, 503):
            raise InferenceError(f"Server error ({response.status_code}): {detail}")
        else:
            raise DebugLLMError(f"Unexpected HTTP {response.status_code}: {detail}")


# ---------------------------------------------------------------------------
# CLI Demo
# ---------------------------------------------------------------------------

def main():
    """Interactive demo — fix a set of example bugs."""

    base_url = os.environ.get("DEBUGLLM_BASE_URL", "http://localhost:8000")
    api_key = os.environ.get("DEBUGLLM_API_KEY", "")

    client = DebugLLMClient(base_url=base_url, api_key=api_key)

    # Health check first
    try:
        health = client.health()
        print(f"\n✅ Service is {health['status']} | Uptime: {health['uptime_seconds']}s")
        print(f"   Redis: {'✓' if health['redis_connected'] else '✗'}  "
              f"vLLM: {'✓' if health['vllm_reachable'] else '✗'}\n")
    except DebugLLMError as e:
        print(f"⚠️  Health check failed: {e}")
        print("   Proceeding anyway...\n")

    examples = [
        ("Missing colon in for loop",
         "for i in range(5)\n    print(i)"),
        ("Index out of range",
         "numbers = [1,2,3,4]\nprint(numbers[4])"),
        ("Missing function indentation",
         "def greet(name):\nprint('Hello, ' + name)"),
        ("Wrong variable name in loop",
         "total = 0\nfor num in range(10):\n    total += number\nprint(total)"),
        ("Missing return statement",
         "def add(a, b):\n    result = a + b"),
    ]

    for title, buggy_code in examples:
        print(f"\n🐛  Bug: {title}")
        print(f"Input:\n{buggy_code}")

        try:
            result = client.fix_bug(buggy_code)
            print(result)
        except DebugLLMError as e:
            print(f"❌ Error: {e}")

        time.sleep(0.5)   # Brief pause between requests


if __name__ == "__main__":
    main()
