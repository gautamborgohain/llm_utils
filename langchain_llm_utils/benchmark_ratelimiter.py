import pytest
from typing import List, Dict
import asyncio
import time
import tiktoken
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_llm_utils.rate_limiter import default_rate_limiter, SmartRateLimiter


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
        enc = tiktoken.encoding_for_model("gpt-4")
        tokens = len(enc.encode(prompt))

        if not self._check_rate_limits(tokens):
            self.rejected_requests += 1
            raise Exception("Rate limit exceeded")

        # Simulate processing time (longer for longer prompts)
        process_time = 0.1 + (
            tokens * 0.01
        )  # Base latency + token-based processing time
        await asyncio.sleep(process_time)  # Use asyncio.sleep instead of time.sleep

        return MockLLMResponse(
            text=f"Processed: {prompt[:10]}...",
            tokens_used=tokens,
            request_time=process_time,
        )


def print_test_metrics(
    label: str,
    duration: float,
    successful_results: int,
    total_prompts: int,
    api_stats: MockLLMAPI,
    successful_tokens: int,
):
    """Print metrics for a single test run"""
    print(f"\nTesting with {label}:")
    print(f"Duration: {duration:.2f}s")
    print(f"Successful requests: {successful_results}/{total_prompts}")
    print(f"Success rate: {(successful_results/total_prompts)*100:.1f}%")
    print(f"Rejected requests: {api_stats.rejected_requests}")
    print(f"Attempted requests: {api_stats.total_requests}")
    print(f"Successful RPM: {successful_results / (duration / 60):.2f}")
    print(f"Successful tokens processed: {successful_tokens}")
    print(f"Successful TPM: {successful_tokens / (duration / 60):.2f}")
    print(f"Attempted RPM: {api_stats.total_requests / (duration / 60):.2f}")
    print(f"Attempted TPM: {api_stats.total_tokens / (duration / 60):.2f}")


def print_comparison_metrics(
    baseline_stats: Dict,
    smart_stats: Dict,
    langchain_stats: Dict,
    total_prompts: int,
    smart_limiter_metrics: Dict,
    is_async: bool = False,
):
    """Print comparison metrics between different rate limiters"""
    async_label = "(async)" if is_async else ""

    # Print smart limiter metrics
    print(f"\nSmart Rate Limiter Metrics {async_label}:")
    for key, value in smart_limiter_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print(f"\nPerformance Comparison {async_label}:")
    print("Success Rates:")
    print(f"  Baseline:   {(baseline_stats['successful']/total_prompts)*100:.1f}%")
    print(f"  Smart:      {(smart_stats['successful']/total_prompts)*100:.1f}%")
    print(f"  LangChain:  {(langchain_stats['successful']/total_prompts)*100:.1f}%")

    print("\nRejected Requests:")
    print(f"  Baseline:   {baseline_stats['rejected']}")
    print(f"  Smart:      {smart_stats['rejected']}")
    print(f"  LangChain:  {langchain_stats['rejected']}")

    print("\nTotal Successful Tokens:")
    print(f"  Baseline:   {baseline_stats['tokens']}")
    print(f"  Smart:      {smart_stats['tokens']}")
    print(f"  LangChain:  {langchain_stats['tokens']}")

    print("\nDuration (seconds):")
    print(f"  Baseline:   {baseline_stats['duration']:.2f}")
    print(f"  Smart:      {smart_stats['duration']:.2f}")
    print(f"  LangChain:  {langchain_stats['duration']:.2f}")

    print("\nSuccessful RPM:")
    print(
        f"  Baseline:   {baseline_stats['successful']/(baseline_stats['duration']/60):.1f}"
    )
    print(f"  Smart:      {smart_stats['successful']/(smart_stats['duration']/60):.1f}")
    print(
        f"  LangChain:  {langchain_stats['successful']/(langchain_stats['duration']/60):.1f}"
    )

    print("\nSuccessful TPM:")
    print(
        f"  Baseline:   {baseline_stats['tokens']/(baseline_stats['duration']/60):.1f}"
    )
    print(f"  Smart:      {smart_stats['tokens']/(smart_stats['duration']/60):.1f}")
    print(
        f"  LangChain:  {langchain_stats['tokens']/(langchain_stats['duration']/60):.1f}"
    )


def run_assertions(baseline_stats: Dict, smart_stats: Dict, langchain_stats: Dict):
    """Run assertions to verify rate limiter performance"""
    assert (
        smart_stats["successful"] >= baseline_stats["successful"]
    ), "Smart rate limiter should improve success rate"
    assert (
        langchain_stats["successful"] >= baseline_stats["successful"]
    ), "LangChain rate limiter should improve success rate"
    assert (
        smart_stats["rejected"] <= baseline_stats["rejected"]
    ), "Smart rate limiter should reduce rejections"
    assert (
        langchain_stats["rejected"] <= baseline_stats["rejected"]
    ), "LangChain rate limiter should reduce rejections"


def collect_test_stats(results: List, duration: float, api: MockLLMAPI) -> Dict:
    """Collect statistics from a test run"""
    successful_results = [r for r in results if isinstance(r, MockLLMResponse)]
    return {
        "successful": len(successful_results),
        "tokens": sum(r.tokens_used for r in successful_results),
        "rejected": api.rejected_requests,
        "duration": duration,
    }


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
    ] * 5  # 15 prompts total, should exceed rate limits

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
    api.total_requests = 0
    api.rejected_requests = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_without_limiter, prompts))

    baseline_stats = collect_test_stats(results, time.time() - start_time, api)
    print_test_metrics(
        "baseline",
        baseline_stats["duration"],
        baseline_stats["successful"],
        len(prompts),
        api,
        baseline_stats["tokens"],
    )

    # Test with smart rate limiter
    api = MockLLMAPI(rpm_limit=60, tpm_limit=4000)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_with_smart_limiter, prompts))

    smart_stats = collect_test_stats(results, time.time() - start_time, api)
    print_test_metrics(
        "smart rate limiter",
        smart_stats["duration"],
        smart_stats["successful"],
        len(prompts),
        api,
        smart_stats["tokens"],
    )

    # Test with LangChain rate limiter
    api = MockLLMAPI(rpm_limit=60, tpm_limit=4000)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_with_langchain_limiter, prompts))

    langchain_stats = collect_test_stats(results, time.time() - start_time, api)
    print_test_metrics(
        "LangChain rate limiter",
        langchain_stats["duration"],
        langchain_stats["successful"],
        len(prompts),
        api,
        langchain_stats["tokens"],
    )

    # Print comparison and run assertions
    print_comparison_metrics(
        baseline_stats,
        smart_stats,
        langchain_stats,
        len(prompts),
        smart_limiter.get_metrics(),
    )
    run_assertions(baseline_stats, smart_stats, langchain_stats)


@pytest.mark.asyncio
async def test_smart_rate_limiter_with_mock_api_async():
    """Test async SmartRateLimiter with a mock LLM API under heavy load

    This test is identical to test_smart_rate_limiter_with_mock_api but uses async methods.
    It tests the aacquire() methods of both rate limiters.
    """
    # Setup
    api = MockLLMAPI(rpm_limit=60, tpm_limit=4000)
    max_concurrent = 10
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
        requests_per_second=1,
        check_every_n_seconds=0.1,
        max_bucket_size=10,
    )

    # Test prompts
    prompts = [
        "Short prompt",
        "This is a medium length prompt that uses more tokens than the short one",
        "This is a much longer prompt that will use even more tokens and really test our rate limiting capabilities with many more words and complex ideas",
    ] * 5

    async def process_with_smart_limiter(prompt: str):
        await smart_limiter.aacquire(prompt)
        try:
            return await api.aprocess_prompt(prompt)
        except Exception:
            return None

    async def process_with_langchain_limiter(prompt: str):
        await langchain_limiter.aacquire()
        try:
            return await api.aprocess_prompt(prompt)
        except Exception:
            return None

    async def process_without_limiter(prompt: str):
        try:
            return await api.aprocess_prompt(prompt)
        except Exception:
            return None

    async def run_concurrent(func, prompts):
        tasks = [func(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)

    # Test without rate limiter (baseline)
    api.total_requests = 0
    api.rejected_requests = 0
    start_time = time.time()

    results = await run_concurrent(process_without_limiter, prompts)

    baseline_stats = collect_test_stats(results, time.time() - start_time, api)
    print_test_metrics(
        "baseline (async)",
        baseline_stats["duration"],
        baseline_stats["successful"],
        len(prompts),
        api,
        baseline_stats["tokens"],
    )

    # Test with smart rate limiter
    api = MockLLMAPI(rpm_limit=60, tpm_limit=4000)
    start_time = time.time()

    results = await run_concurrent(process_with_smart_limiter, prompts)

    smart_stats = collect_test_stats(results, time.time() - start_time, api)
    print_test_metrics(
        "smart rate limiter (async)",
        smart_stats["duration"],
        smart_stats["successful"],
        len(prompts),
        api,
        smart_stats["tokens"],
    )

    # Test with LangChain rate limiter
    api = MockLLMAPI(rpm_limit=60, tpm_limit=4000)
    start_time = time.time()

    results = await run_concurrent(process_with_langchain_limiter, prompts)

    langchain_stats = collect_test_stats(results, time.time() - start_time, api)
    print_test_metrics(
        "LangChain rate limiter (async)",
        langchain_stats["duration"],
        langchain_stats["successful"],
        len(prompts),
        api,
        langchain_stats["tokens"],
    )

    # Print comparison and run assertions
    print_comparison_metrics(
        baseline_stats,
        smart_stats,
        langchain_stats,
        len(prompts),
        smart_limiter.get_metrics(),
        is_async=True,
    )
    run_assertions(baseline_stats, smart_stats, langchain_stats)
