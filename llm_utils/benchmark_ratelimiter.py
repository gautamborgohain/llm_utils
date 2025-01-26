"""
Expected stdout:
-----------------
tests/test_rate_limiter.py::test_smart_rate_limiter_with_mock_api
Testing without rate limiter (baseline):
Duration: 1.70s
Successful requests: 60/150
Success rate: 40.0%
Rejected requests: 90
Attempted requests: 60
Successful RPM: 2111.68
Successful tokens processed: 840
Successful TPM: 29563.53
Attempted RPM: 2111.68
Attempted TPM: 29563.53

Testing with smart rate limiter:
Duration: 153.22s
Successful requests: 150/150
Success rate: 100.0%
Rejected requests: 0
Attempted requests: 150
Successful RPM: 58.74
Successful tokens processed: 2100
Successful TPM: 822.36
Attempted RPM: 58.74
Attempted TPM: 822.36

Testing with LangChain rate limiter:
Duration: 150.37s
Successful requests: 147/150
Success rate: 98.0%
Rejected requests: 3
Attempted requests: 147
Successful RPM: 58.65
Successful tokens processed: 2070
Successful TPM: 825.94
Attempted RPM: 58.65
Attempted TPM: 825.94

Smart Rate Limiter Metrics:
available_requests: 1.00
available_tokens: 66.67
requests_per_minute: 60.00
tokens_per_minute: 4000.00
queue_length: 0
total_requests_processed: 150
total_tokens_processed: 2100
active_tasks: 0

Performance Comparison:
Success Rates:
Baseline:   40.0%
Smart:      100.0%
LangChain:  98.0%

Rejected Requests:
Baseline:   90
Smart:      0
LangChain:  3

Total Successful Tokens:
Baseline:   840
Smart:      2100
LangChain:  2070

Duration (seconds):
Baseline:   1.70
Smart:      153.22
LangChain:  150.37

Successful RPM:
Baseline:   2111.7
Smart:      58.7
LangChain:  58.7

Successful TPM:
Baseline:   29563.5
Smart:      822.4
LangChain:  825.9
PASSED
"""

from typing import List, Dict
import asyncio
import time
import tiktoken
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from langchain_core.rate_limiters import InMemoryRateLimiter
from llm_utils.rate_limiter import default_rate_limiter, SmartRateLimiter


@dataclass
class MockLLMResponse:
    text: str
    tokens_used: int
    request_time: float


class MockLLMAPI:
    """Simulates an LLM API with rate limits and processing delays"""

    def __init__(self, rpm_limit: int, tpm_limit: int):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self._request_times = []
        self._token_times = []
        self._lock = threading.Lock()
        self.total_requests = 0
        self.total_tokens = 0
        self.rejected_requests = 0

    def _check_rate_limits(self, tokens: int) -> bool:
        """Check if the request would exceed rate limits"""
        now = time.time()
        minute_ago = now - 60

        with self._lock:
            # Clean up old entries
            self._request_times = [t for t in self._request_times if t > minute_ago]
            self._token_times = [t for t in self._token_times if t > minute_ago]

            # Check limits
            if len(self._request_times) >= self.rpm_limit:
                return False
            if sum(self._token_times) + tokens >= self.tpm_limit:
                return False

            # Update tracking
            self._request_times.append(now)
            self._token_times.append(tokens)
            self.total_requests += 1
            self.total_tokens += tokens
            return True

    def process_prompt(self, prompt: str) -> MockLLMResponse:
        """Process a prompt, respecting rate limits

        We first check if the request would exceed rate limits. If it does, we raise an exception.
        If it doesn't, we simulate processing time which is linear with the number of tokens in the prompt.

        """
        enc = tiktoken.encoding_for_model("gpt-4")
        tokens = len(enc.encode(prompt))

        if not self._check_rate_limits(tokens):
            self.rejected_requests += 1
            raise Exception("Rate limit exceeded")

        # Simulate processing time (longer for longer prompts)
        process_time = 0.1 + (
            tokens * 0.01
        )  # Base latency + token-based processing time
        time.sleep(process_time)

        return MockLLMResponse(
            text=f"Processed: {prompt[:10]}...",
            tokens_used=tokens,
            request_time=process_time,
        )

    async def aprocess_prompt(self, prompt: str) -> MockLLMResponse:
        """Async version of process_prompt"""
        return await asyncio.to_event_loop().run_in_executor(
            None, self.process_prompt, prompt
        )


def test_smart_rate_limiter_with_mock_api():
    """Test SmartRateLimiter with a mock LLM API under heavy load

    MockLLM API limits:
    - 60 requests per minute
    - 4000 tokens per minute

    Test processing 150 prompts of varying lengths with 10 workers
    This is to simulate a load higher than the LLM API can handle
    - We are making 10 requests per second to the API that can only handle 1 request per second.
    - If a request to the API fails (i.e we exceed rate limits), we count it as a rejected request.
    - We then compare the success rate, rejected requests, and processing time of the different rate limiters.

    We compare 3 scenarios:
    - Baseline: No rate limiter
    - Smart rate limiter (using requests and tokens)
    - LangChain rate limiter (In Memory Rate Limiter using only requests to control)

    Ideally, the smart rate limiter should improve the success rate and keep the RPM and TPM close to the API limits.
    """
    # Setup
    api = MockLLMAPI(rpm_limit=60, tpm_limit=4000)  # 1 request/sec, ~66 tokens/sec
    max_workers = 10
    enc = tiktoken.encoding_for_model("gpt-4")
    tokenizer = lambda x: len(enc.encode(x))

    smart_limiter = SmartRateLimiter(
        requests_per_minute=60,
        tokens_per_minute=4000,
        max_request_burst=10,
        tokenizer=tokenizer,
        check_every_n_seconds=0.1,
    )

    langchain_limiter = InMemoryRateLimiter(
        requests_per_second=1,  # <-- Can only make a request once every 10 seconds!!
        check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
        max_bucket_size=10,  # Controls the maximum burst size.
    )

    # Test prompts of varying lengths - increased volume and variety
    prompts = [
        "Short prompt",  # ~2 tokens
        "This is a medium length prompt that uses more tokens than the short one",  # ~12 tokens
        "This is a much longer prompt that will use even more tokens and really test our rate limiting capabilities with many more words and complex ideas",  # ~27 tokens
    ] * 50  # 150 prompts total, should exceed rate limits

    def process_with_smart_limiter(prompt: str):
        smart_limiter.acquire(prompt)
        try:
            return api.process_prompt(prompt)
        except Exception:
            return None

    def process_with_langchain_limiter(prompt: str):
        langchain_limiter.acquire()
        try:
            return api.process_prompt(prompt)
        except Exception:
            return None

    def process_without_limiter(prompt: str):
        try:
            return api.process_prompt(prompt)
        except Exception:
            return None

    # Test without rate limiter (baseline)
    print("\nTesting without rate limiter (baseline):")
    api.total_requests = 0
    api.rejected_requests = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_without_limiter, prompts))

    baseline_duration = time.time() - start_time
    successful_results = [r for r in results if r is not None]
    successful_baseline = len(successful_results)
    baseline_tokens = sum(r.tokens_used for r in successful_results)
    baseline_rejected = api.rejected_requests

    print(f"Duration: {baseline_duration:.2f}s")
    print(f"Successful requests: {successful_baseline}/{len(prompts)}")
    print(f"Success rate: {(successful_baseline/len(prompts))*100:.1f}%")
    print(f"Rejected requests: {api.rejected_requests}")
    print(f"Attempted requests: {api.total_requests}")
    print(f"Successful RPM: {successful_baseline / (baseline_duration / 60):.2f}")
    print(f"Successful tokens processed: {baseline_tokens}")
    print(f"Successful TPM: {baseline_tokens / (baseline_duration / 60):.2f}")
    print(f"Attempted RPM: {api.total_requests / (baseline_duration / 60):.2f}")
    print(f"Attempted TPM: {api.total_tokens / (baseline_duration / 60):.2f}")

    # Test with smart rate limiter
    api = MockLLMAPI(rpm_limit=60, tpm_limit=4000)
    print("\nTesting with smart rate limiter:")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_with_smart_limiter, prompts))

    smart_duration = time.time() - start_time
    successful_results = [r for r in results if r is not None]
    successful_smart = len(successful_results)
    smart_tokens = sum(r.tokens_used for r in successful_results)
    smart_rejected = api.rejected_requests

    print(f"Duration: {smart_duration:.2f}s")
    print(f"Successful requests: {successful_smart}/{len(prompts)}")
    print(f"Success rate: {(successful_smart/len(prompts))*100:.1f}%")
    print(f"Rejected requests: {api.rejected_requests}")
    print(f"Attempted requests: {api.total_requests}")
    print(f"Successful RPM: {successful_smart / (smart_duration / 60):.2f}")
    print(f"Successful tokens processed: {smart_tokens}")
    print(f"Successful TPM: {smart_tokens / (smart_duration / 60):.2f}")
    print(f"Attempted RPM: {api.total_requests / (smart_duration / 60):.2f}")
    print(f"Attempted TPM: {api.total_tokens / (smart_duration / 60):.2f}")

    # Test with LangChain rate limiter
    api = MockLLMAPI(rpm_limit=60, tpm_limit=4000)
    print("\nTesting with LangChain rate limiter:")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_with_langchain_limiter, prompts))

    langchain_duration = time.time() - start_time
    successful_results = [r for r in results if r is not None]
    successful_langchain = len(successful_results)
    langchain_tokens = sum(r.tokens_used for r in successful_results)
    langchain_rejected = api.rejected_requests  # Capture rejected count

    print(f"Duration: {langchain_duration:.2f}s")
    print(f"Successful requests: {successful_langchain}/{len(prompts)}")
    print(f"Success rate: {(successful_langchain/len(prompts))*100:.1f}%")
    print(f"Rejected requests: {api.rejected_requests}")
    print(f"Attempted requests: {api.total_requests}")
    print(f"Successful RPM: {successful_langchain / (langchain_duration / 60):.2f}")
    print(f"Successful tokens processed: {langchain_tokens}")
    print(f"Successful TPM: {langchain_tokens / (langchain_duration / 60):.2f}")
    print(f"Attempted RPM: {api.total_requests / (langchain_duration / 60):.2f}")
    print(f"Attempted TPM: {api.total_tokens / (langchain_duration / 60):.2f}")

    # Get smart limiter metrics
    metrics = smart_limiter.get_metrics()
    print("\nSmart Rate Limiter Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    # Performance comparison
    print("\nPerformance Comparison:")
    print("Success Rates:")
    print(f"  Baseline:   {(successful_baseline/len(prompts))*100:.1f}%")
    print(f"  Smart:      {(successful_smart/len(prompts))*100:.1f}%")
    print(f"  LangChain:  {(successful_langchain/len(prompts))*100:.1f}%")

    print("\nRejected Requests:")
    print(f"  Baseline:   {baseline_rejected}")
    print(f"  Smart:      {smart_rejected}")
    print(f"  LangChain:  {langchain_rejected}")

    print("\nTotal Successful Tokens:")
    print(f"  Baseline:   {baseline_tokens}")
    print(f"  Smart:      {smart_tokens}")
    print(f"  LangChain:  {langchain_tokens}")

    print("\nDuration (seconds):")
    print(f"  Baseline:   {baseline_duration:.2f}")
    print(f"  Smart:      {smart_duration:.2f}")
    print(f"  LangChain:  {langchain_duration:.2f}")

    print("\nSuccessful RPM:")
    print(f"  Baseline:   {successful_baseline/(baseline_duration/60):.1f}")
    print(f"  Smart:      {successful_smart/(smart_duration/60):.1f}")
    print(f"  LangChain:  {successful_langchain/(langchain_duration/60):.1f}")

    print("\nSuccessful TPM:")
    print(f"  Baseline:   {baseline_tokens/(baseline_duration/60):.1f}")
    print(f"  Smart:      {smart_tokens/(smart_duration/60):.1f}")
    print(f"  LangChain:  {langchain_tokens/(langchain_duration/60):.1f}")

    # Assertions
    assert (
        successful_smart >= successful_baseline
    ), "Smart rate limiter should improve success rate"
    assert (
        successful_langchain >= successful_baseline
    ), "LangChain rate limiter should improve success rate"
    assert (
        smart_rejected <= baseline_rejected
    ), "Smart rate limiter should reduce rejections"
    assert (
        langchain_rejected <= baseline_rejected
    ), "LangChain rate limiter should reduce rejections"
