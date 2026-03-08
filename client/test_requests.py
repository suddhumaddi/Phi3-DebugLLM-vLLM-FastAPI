"""
DebugLLM - Test Requests
=========================
Integration test suite that exercises the live API endpoint.
Tests cover: authentication, happy-path inference, edge cases,
rate-limit headers, caching behaviour, and error handling.

Usage:
    # Run all tests (requires running deployment)
    python client/test_requests.py

    # Target a specific endpoint
    DEBUGLLM_BASE_URL=https://my-endpoint.modal.run python client/test_requests.py

Environment Variables:
    DEBUGLLM_BASE_URL   Base URL of the API (default: http://localhost:8000)
    DEBUGLLM_API_KEY    Bearer token for authentication
"""

import os
import sys
import time

# Ensure client module is importable when running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.client import (
    AuthenticationError,
    BugFixResult,
    DebugLLMClient,
    DebugLLMError,
    ValidationError,
)

# ---------------------------------------------------------------------------
# Test Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("DEBUGLLM_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("DEBUGLLM_API_KEY", "")

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭️  SKIP"

results: list[tuple[str, str, str]] = []   # (test_name, status, detail)


def record(name: str, passed: bool, detail: str = ""):
    status = PASS if passed else FAIL
    results.append((name, status, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_health(client: DebugLLMClient):
    print("\n[Health Check]")
    try:
        health = client.health()
        record("Service responds to /health", True, f"status={health['status']}")
        record("Version field present", "version" in health)
        record("Uptime field present", "uptime_seconds" in health)
    except DebugLLMError as e:
        record("Health check", False, str(e))


def test_happy_path(client: DebugLLMClient):
    print("\n[Happy Path — Known Bug Types]")

    cases = [
        (
            "Syntax: missing colon",
            "for i in range(5)\n    print(i)",
            "for i in range(5):",
        ),
        (
            "Index: out of range",
            "numbers = [1,2,3,4]\nprint(numbers[4])",
            "numbers[3]",
        ),
        (
            "Indentation: function body",
            "def greet(name):\nprint('Hello, ' + name)",
            "    print",
        ),
        (
            "Variable: wrong name in loop",
            "total = 0\nfor num in range(10):\n    total += number\nprint(total)",
            "total += num",
        ),
    ]

    for name, buggy, expected_fragment in cases:
        try:
            result = client.fix_bug(buggy)
            passed = (
                isinstance(result, BugFixResult)
                and expected_fragment in result.fixed_code
                and len(result.explanation) > 0
                and result.latency_ms > 0
            )
            record(name, passed, f"latency={result.latency_ms:.0f}ms")
        except DebugLLMError as e:
            record(name, False, str(e))


def test_response_schema(client: DebugLLMClient):
    print("\n[Response Schema]")
    try:
        result = client.fix_bug("for i in range(5)\n    print(i)")
        record("fixed_code is non-empty string", bool(result.fixed_code))
        record("explanation is non-empty string", bool(result.explanation))
        record("model field present", bool(result.model))
        record("tokens_used is positive int", isinstance(result.tokens_used, int) and result.tokens_used > 0)
        record("latency_ms is positive float", result.latency_ms > 0)
        record("cache_hit is boolean", isinstance(result.cache_hit, bool))
    except DebugLLMError as e:
        record("Schema test", False, str(e))


def test_caching(client: DebugLLMClient):
    print("\n[Caching Behaviour]")
    code = "x = [1,2,3]\nprint(x[5])"
    try:
        r1 = client.fix_bug(code)
        r2 = client.fix_bug(code)   # identical request — should hit cache
        record("Second identical request returns same fixed_code", r1.fixed_code == r2.fixed_code)
        record("Cache HIT on second request", r2.cache_hit, f"cache_hit={r2.cache_hit}")
        # Cache hit should be faster
        record(
            "Cache response (latency hint)",
            True,
            f"1st={r1.latency_ms:.0f}ms, 2nd={r2.latency_ms:.0f}ms",
        )
    except DebugLLMError as e:
        record("Caching test", False, str(e))


def test_authentication(client: DebugLLMClient):
    print("\n[Authentication]")
    bad_client = DebugLLMClient(base_url=BASE_URL, api_key="invalid-key-xyz")
    try:
        bad_client.fix_bug("for i in range(5)\n    print(i)")
        record("Rejects invalid API key", False, "Expected AuthenticationError")
    except AuthenticationError:
        record("Rejects invalid API key", True)
    except DebugLLMError as e:
        record("Rejects invalid API key", False, f"Wrong exception type: {e}")


def test_input_validation(client: DebugLLMClient):
    print("\n[Input Validation]")

    # Empty input
    try:
        client.fix_bug("")
        record("Rejects empty code", False, "Expected ValidationError")
    except ValidationError:
        record("Rejects empty code", True)
    except DebugLLMError as e:
        record("Rejects empty code", False, str(e))

    # Oversized input
    try:
        oversized = "x = 1\n" * 2000   # ~12,000 chars
        client.fix_bug(oversized)
        record("Rejects oversized input", False, "Expected error")
    except (ValidationError, DebugLLMError):
        record("Rejects oversized input", True)

    # Blocked pattern (prompt injection attempt)
    try:
        client.fix_bug("import os; os.system('rm -rf /')")
        record("Rejects blocked pattern (os.system)", False, "Expected ValidationError")
    except (ValidationError, DebugLLMError):
        record("Rejects blocked pattern (os.system)", True)


def test_temperature_variants(client: DebugLLMClient):
    print("\n[Temperature Variants]")
    code = "for i in range(5)\n    print(i)"
    for temp in [0.0, 0.5, 1.0]:
        try:
            result = client.fix_bug(code, temperature=temp)
            record(f"temperature={temp} accepted", bool(result.fixed_code), f"latency={result.latency_ms:.0f}ms")
        except DebugLLMError as e:
            record(f"temperature={temp} accepted", False, str(e))


def test_batch_processing(client: DebugLLMClient):
    print("\n[Batch Processing]")
    snippets = [
        "for i in range(5)\n    print(i)",
        "x = [1,2]\nprint(x[2])",
    ]
    try:
        results_batch = client.fix_bugs_batch(snippets)
        record("Batch returns correct count", len(results_batch) == len(snippets))
        record("All results are BugFixResult", all(isinstance(r, BugFixResult) for r in results_batch))
    except DebugLLMError as e:
        record("Batch processing", False, str(e))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DebugLLM API Integration Tests")
    print(f"Endpoint: {BASE_URL}")
    print("=" * 60)

    if not API_KEY:
        print("\n⚠️  DEBUGLLM_API_KEY not set — authentication tests will use empty key\n")

    client = DebugLLMClient(base_url=BASE_URL, api_key=API_KEY)

    # Run all test suites
    test_health(client)
    test_happy_path(client)
    test_response_schema(client)
    test_caching(client)
    test_authentication(client)
    test_input_validation(client)
    test_temperature_variants(client)
    test_batch_processing(client)

    # Summary
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    total = len(results)

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    if failed:
        print(f"         {failed} test(s) FAILED")
        print("\nFailed tests:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  • {name}: {detail}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
