import asyncio
import heapq
import time
import threading
import tiktoken
from llm_utils.config import config
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

    2. Priority Queue Management
    - Requests are queued when capacity isn't immediately available
    - Priority calculation:
        * 80% weight: Wait time (prevents starvation)
        * 20% weight: Request size (favors smaller requests for better throughput)
    - Queue is continuously processed asynchronously
    - Priorities are recalculated dynamically as wait times increase

    3. Token Management
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
        self._pending_queue: List[tuple[int, float, float]] = (
            []
        )  # [(tokens, timestamp, priority)]
        self._running = True
        self._queue_processor: Optional[asyncio.Task] = None
        self._active_tasks: Set[asyncio.Task] = set()

        # Statistics
        self._stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "wait_times": [],  # [(timestamp, wait_time)]
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

    def _calculate_priority(self, token_count: int, wait_time: float) -> float:
        """Calculate request priority based on size and wait time."""
        # Normalize values
        max_tokens = self.token_bucket.max_capacity
        max_wait = 60.0  # 1 minute reference

        norm_size = min(token_count / max_tokens, 1.0)
        norm_wait = min(wait_time / max_wait, 1.0)

        # Prioritize wait time (60%) over size (40%)
        return (self.wait_time_weight * norm_wait) + (
            self.size_weight * (1 - norm_size)
        )

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

        while self._running:
            if self._can_process(token_count):
                return True
            time.sleep(self.check_every_n_seconds)

        return False

    async def aacquire(
        self, text: Optional[str] = None, *, blocking: bool = True
    ) -> bool:
        """Asynchronously attempt to acquire capacity for a request."""
        token_count = self._count_tokens(text)

        if self._can_process(token_count):
            return True

        if not blocking:
            return False

        # Add to priority queue
        entry = (token_count, time.monotonic(), 0.0)
        with self._lock:
            self._pending_queue.append(entry)

            # Start queue processor if needed
            if not self._queue_processor or self._queue_processor.done():
                self._queue_processor = asyncio.create_task(self._process_queue())
                self._active_tasks.add(self._queue_processor)

        # Wait for processing
        while self._running:
            if entry not in self._pending_queue:
                return True
            await asyncio.sleep(self.check_every_n_seconds)

        return False

    async def _process_queue(self) -> None:
        """Process pending requests based on priority."""
        while self._running and self._pending_queue:
            with self._lock:
                now = time.monotonic()

                # Update priorities
                for i, (tokens, timestamp, _) in enumerate(self._pending_queue):
                    wait_time = now - timestamp
                    priority = self._calculate_priority(tokens, wait_time)
                    self._pending_queue[i] = (tokens, timestamp, priority)

                # Sort by priority
                self._pending_queue.sort(key=lambda x: (-x[2], x[1]))

                # Try to process highest priority
                if self._pending_queue and self._can_process(self._pending_queue[0][0]):
                    entry = self._pending_queue.pop(0)
                    self._stats["wait_times"].append((now, now - entry[1]))

            await asyncio.sleep(self.check_every_n_seconds)

    async def shutdown(self) -> None:
        """Gracefully shutdown the rate limiter."""
        self._running = False
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics about rate limiter state."""
        with self._lock:
            self._update_buckets()

            metrics = {
                "available_requests": self.request_bucket.available_tokens,
                "available_tokens": self.token_bucket.available_tokens,
                "requests_per_minute": self.requests_per_second * 60,
                "tokens_per_minute": self.tokens_per_second * 60,
                "queue_length": len(self._pending_queue),
                "total_requests_processed": self._stats["total_requests"],
                "total_tokens_processed": self._stats["total_tokens"],
                "active_tasks": len(self._active_tasks),
            }

            # Add wait time stats if available
            recent_waits = [wt for _, wt in self._stats["wait_times"][-100:]]
            if recent_waits:
                metrics.update(
                    {
                        "average_wait_time": sum(recent_waits) / len(recent_waits),
                        "max_wait_time": max(recent_waits),
                        "min_wait_time": min(recent_waits),
                    }
                )

            return metrics
