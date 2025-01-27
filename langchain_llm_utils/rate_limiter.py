import asyncio
import heapq
import time
import threading
import tiktoken
from langchain_llm_utils.config import config
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Set
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.rate_limiters import InMemoryRateLimiter

default_rate_limiter = InMemoryRateLimiter(
    requests_per_second=config.default_rate_limiter_requests_per_second,
    check_every_n_seconds=config.default_rate_limiter_check_every_n_seconds,
    max_bucket_size=config.default_rate_limiter_max_bucket_size,
)


class Tokenizer(Protocol):
    """Protocol for tokenizers that can count tokens in text."""

    def __call__(self, text: str) -> int:
        """Count tokens in the given text."""
        ...


class GPT4Tokenizer(Tokenizer):
    """Tokenizer for GPT-4 that can count tokens in text."""

    def __init__(self):
        self.enc = tiktoken.encoding_for_model("gpt-4")

    def __call__(self, text: str) -> int:
        """Count tokens in the given text."""
        return len(self.enc.encode(text))


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    available_tokens: float
    last_update: float
    max_capacity: float


class SmartRateLimiter(BaseRateLimiter):
    """
    Smart Rate Limiter for Token-Aware API Request Management

    This module implements an advanced rate limiter designed specifically for LLM API requests
    where both request count and token count need to be managed.
    It uses a dual token bucket algorithm combined with priority-based queuing to
    maximize throughput while maintaining strict rate limits.

    About token bucket algorithm:
    The bucket is filled with tokens at a given rate.
    Each request consumes a token. If there are not enough tokens in the bucket,
    the request is blocked until there are enough tokens.

    Algorithm Overview
    ------------------
    1. Dual Token Bucket System
    - Request Bucket: Controls requests/minute
    - Token Bucket: Controls tokens/minute
    - Each bucket refills continuously at its specified rate
    - Burst capacity is configurable for both buckets

    2. Token Management
    - Integrated tokenizer for accurate token counting
    - Historical statistics for adaptive behavior
    - Fallback strategies for unknown token counts (100 tokens default)
    - Rolling statistics window for token usage patterns

    How it works
    ------------------
    1. New Request Flow:
    - Count tokens in the request
    - Attempt immediate processing
    - Queue if capacity unavailable

    2. Continuous Processing:
    - Update token buckets based on elapsed time
    - Recalculate request priorities
    - Process highest priority requests first
    - Adaptive sleep intervals based on queue state

    3. Success/Failure Conditions:
    ```
    if (request_bucket.available_tokens >= 1 AND
        token_bucket.available_tokens >= required_tokens):
        # Consume tokens
        # Return Success
    else:
        # Queue or return failure based on blocking mode
    ```

    Example Usage
    ------------------
    Using with GPT4Tokenizer (tiktoken)
    ```
    from utils.llm_utils import GPT4Tokenizer
    tokenizer = GPT4Tokenizer()

    limiter = SmartRateLimiter(
        requests_per_minute=60,
        tokens_per_minute=40_000,
        tokenizer=tokenizer
    )
    ```

    Use with LangChain
    ```
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(
        model_name="gpt-4",
        rate_limiter=limiter
    )
    ```

    Use with custom LLM services
    ```
    from services.translation.models import GenericTranslationModel
    tm = GenericTranslationModel(
        name="ollama:llama3.1",
        rate_limiter=rate_limiter,
    )
    tm.predict(
        text="The capital of France is Paris.",
        target_language="French",
    )
    ```

    Example Scenario
    ------------------
    Consider a system configured for 60 requests/minute and 40,000 tokens/minute:

    Initial state: Both buckets full

    Request sequence:
    ```
    t=0.0s: Request A (500 tokens)  → Processed immediately
    t=0.1s: Request B (800 tokens)  → Processed immediately
    t=0.2s: Request C (1000 tokens) → Queued (insufficient tokens)
    t=0.5s: Request D (300 tokens)  → Queued but higher priority than C
    ```

    At t=1.0s: Buckets refill with 1s worth of capacity
    - D processed first (smaller size, decent wait time)
    - C processed when enough tokens accumulate

    """

    def __init__(
        self,
        *,
        requests_per_minute: float = 60,
        tokens_per_minute: float = 40000,
        tokenizer: Tokenizer,
        check_every_n_seconds: float = 0.1,
        wait_time_weight: float = 0.6,
        size_weight: float = 0.4,
        max_request_burst: Optional[float] = None,
        max_token_burst: Optional[float] = None,
    ) -> None:
        """Initialize the smart rate limiter.

        Args:
            requests_per_minute: Number of requests allowed per minute
            tokens_per_minute: Number of tokens allowed per minute
            tokenizer: Tokenizer instance for counting tokens
            check_every_n_seconds: How often to check for available capacity
            max_request_burst: Maximum requests that can be processed in a burst
            max_token_burst: Maximum tokens that can be processed in a burst
        """
        # Convert to per-second rates
        self.requests_per_second = requests_per_minute / 60.0
        self.tokens_per_second = tokens_per_minute / 60.0
        self.tokenizer = tokenizer
        self.check_every_n_seconds = check_every_n_seconds
        self.wait_time_weight = wait_time_weight
        self.size_weight = size_weight

        # Initialize buckets
        now = time.monotonic()
        self.request_bucket = TokenBucket(
            available_tokens=max_request_burst or requests_per_minute,
            last_update=now,
            max_capacity=max_request_burst or requests_per_minute,
        )
        self.token_bucket = TokenBucket(
            available_tokens=max_token_burst or tokens_per_minute,
            last_update=now,
            max_capacity=max_token_burst or tokens_per_minute,
        )

        # Thread safety
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "request_sizes": [],  # [(timestamp, token_count)]
        }

    def _update_buckets(self) -> None:
        """Update token buckets based on elapsed time."""
        now = time.monotonic()

        # Update request bucket
        elapsed = now - self.request_bucket.last_update
        self.request_bucket.available_tokens = min(
            self.request_bucket.max_capacity,
            self.request_bucket.available_tokens + (elapsed * self.requests_per_second),
        )
        self.request_bucket.last_update = now

        # Update token bucket
        elapsed = now - self.token_bucket.last_update
        self.token_bucket.available_tokens = min(
            self.token_bucket.max_capacity,
            self.token_bucket.available_tokens + (elapsed * self.tokens_per_second),
        )
        self.token_bucket.last_update = now

    def _can_process(self, token_count: int) -> bool:
        """Check if there's capacity to process a request."""
        with self._lock:
            self._update_buckets()

            if (
                self.request_bucket.available_tokens >= 1
                and self.token_bucket.available_tokens >= token_count
            ):
                self.request_bucket.available_tokens -= 1
                self.token_bucket.available_tokens -= token_count
                self._stats["total_requests"] += 1
                self._stats["total_tokens"] += token_count
                return True
            return False

    def _count_tokens(self, text: Optional[str] = None) -> int:
        """Count tokens in text or estimate if not provided."""
        if text is None:
            # Use historical average or default
            recent_sizes = [size for _, size in self._stats["request_sizes"][-100:]]
            return int(sum(recent_sizes) / len(recent_sizes)) if recent_sizes else 100

        try:
            count = self.tokenizer(text)
            self._stats["request_sizes"].append((time.monotonic(), count))
            return count
        except Exception:
            return 100  # Conservative fallback

    def acquire(self, text: Optional[str] = None, *, blocking: bool = True) -> bool:
        """Attempt to acquire capacity for a request."""
        token_count = self._count_tokens(text)

        if self._can_process(token_count):
            return True

        if not blocking:
            return False

        while not self._can_process(token_count):
            time.sleep(self.check_every_n_seconds)

        return True

    async def aacquire(
        self, text: Optional[str] = None, *, blocking: bool = True
    ) -> bool:
        """Asynchronously attempt to acquire capacity for a request."""
        token_count = self._count_tokens(text)

        if not blocking:
            return self._can_process(token_count)

        while not self._can_process(token_count):
            await asyncio.sleep(self.check_every_n_seconds)
        return True

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics about rate limiter state."""
        with self._lock:
            self._update_buckets()

            metrics = {
                "available_requests": self.request_bucket.available_tokens,
                "available_tokens": self.token_bucket.available_tokens,
                "requests_per_minute": self.requests_per_second * 60,
                "tokens_per_minute": self.tokens_per_second * 60,
                "total_requests_processed": self._stats["total_requests"],
                "total_tokens_processed": self._stats["total_tokens"],
            }

            return metrics
