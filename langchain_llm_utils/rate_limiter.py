import asyncio
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.rate_limiters import InMemoryRateLimiter
from uuid import UUID
import tiktoken
from langchain_llm_utils.config import config
from langchain_llm_utils.common import get_logger

logger = get_logger("RateLimiter")

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
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    def __call__(self, text: str) -> int:
        """Count tokens in the given text."""
        return len(self.enc.encode(text))


@dataclass
class TokenBucket:
    """
    Token bucket with timestamp for rate limiting.

    Attributes:
        tokens: Current number of available tokens
        last_update: Timestamp of last bucket update
        max_tokens: Maximum token capacity (burst limit)
    """

    tokens: float
    last_update: float
    max_tokens: float


class SmartRateLimiter(BaseRateLimiter):
    """
    Smart Rate Limiter for Token-Aware API Request Management

    This module implements an advanced rate limiter designed specifically for LLM API
    requests where both request count and token count need to be managed. It uses a
    dual token bucket algorithm combined with priority-based queuing to maximize
    throughput while maintaining strict rate limits.

    About token bucket algorithm: The bucket is filled with tokens at a given rate.
    Each request consumes a token. If there are not enough tokens in the bucket,
    the request is blocked until there are enough tokens.

    Algorithm Overview:
    ------------------
    1. Dual Token Bucket System
    - Request Bucket: Controls requests/minute
    - Token Bucket: Controls tokens/minute
    - Each bucket refills continuously at its specified rate
    - Burst capacity is configurable for both buckets

    2. Token Management
    - Integrated tokenizer for accurate token counting
    - Historical statistics for adaptive behavior
    - Fallback strategies for unknown token counts

    How it works:
    ----------------
    1. New Request → Count Tokens → Try Immediate Processing
    2. While Running:
        - Update Token Buckets
        - Sleep for Check Interval
    3. If (request_bucket.tokens ≥ 1 AND token_bucket.tokens ≥ required_tokens):
        - Consume Tokens
        - Return Success
    Else:
        - Return Failure

    Example Usage:
    -------------
    ```python
    # Using with GPT4Tokenizer (tiktoken)

    from utils.llm_utils import GPT4Tokenizer
    tokenizer = GPT4Tokenizer()

    limiter = SmartRateLimiter(
        requests_per_minute=60,
        tokens_per_minute=40_000,
        tokenizer=tokenizer
    )

    # Use with LangChain
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(
        model_name="gpt-4",
        rate_limiter=limiter
    )

    # Use with Lyra service
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

    Example Scenario:
    ----------------
    - System configured for 60 requests/minute and 40,000 tokens/minute
    - Initial state: Both buckets full
    - We send requests in sequnce:
        t=0.0s: Request A (500 tokens)  → Processed immediately
        t=0.1s: Request B (800 tokens)  → Processed immediately
        t=0.2s: Request C (1000 tokens) → Queued (insufficient tokens)
        t=0.5s: Request D (300 tokens)  → Queued but higher priority than C
    - At t=1.0s: Buckets refill with 1s worth of capacity
        - D processed first (smaller size, decent wait time)
        - C processed when enough tokens accumulate
    """

    def __init__(
        self,
        requests_per_minute: float,
        tokens_per_minute: float,
        tokenizer: Tokenizer,
        max_request_burst: Optional[float] = None,
        max_token_burst: Optional[float] = None,
        check_every_n_seconds: float = 0.1,
        stats_window_seconds: float = 3600,  # 1 hour
    ):
        """
        Initialize the smart rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            tokens_per_minute: Maximum tokens per minute
            tokenizer: Callable that counts tokens in text
            max_request_burst: Maximum burst size for requests
            max_token_burst: Maximum burst size for tokens
            check_every_n_seconds: Frequency to check if we can consume tokens
            stats_window_seconds: How long to keep historical stats
        """
        self.requests_per_second = requests_per_minute / 60.0
        self.tokens_per_second = tokens_per_minute / 60.0
        self.tokenizer = tokenizer
        self.check_every_n_seconds = check_every_n_seconds
        self.stats_window_seconds = stats_window_seconds
        self.callback: Optional[RateLimitCallback] = None

        # Initialize token buckets
        now = time.monotonic()
        self.request_bucket = TokenBucket(
            tokens=max_request_burst or requests_per_minute / 60.0,
            last_update=now,
            max_tokens=max_request_burst or requests_per_minute / 60.0,
        )
        self.token_bucket = TokenBucket(
            tokens=max_token_burst or tokens_per_minute / 60.0,
            last_update=now,
            max_tokens=max_token_burst or tokens_per_minute / 60.0,
        )

        # Thread safety
        self._lock = threading.Lock()

        # Stats with timestamps for rolling window
        self._stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "request_sizes": [],  # List[Tuple[float, int]]  # (timestamp, size)
        }

    def _update_buckets(self) -> None:
        """
        Update both token buckets based on elapsed time.

        Implements a minimum update interval to prevent excessive updates.
        """
        now = time.monotonic()

        # Update request bucket
        elapsed = now - self.request_bucket.last_update
        self.request_bucket.tokens = min(
            self.request_bucket.max_tokens,
            self.request_bucket.tokens + (elapsed * self.requests_per_second),
        )
        self.request_bucket.last_update = now

        # Update token bucket
        elapsed = now - self.token_bucket.last_update
        self.token_bucket.tokens = min(
            self.token_bucket.max_tokens,
            self.token_bucket.tokens + (elapsed * self.tokens_per_second),
        )
        self.token_bucket.last_update = now

    def _can_consume(self, token_count: int) -> bool:
        """
        Check if we have capacity for both request and tokens.

        This method is atomic - if it returns True, it has already
        consumed the necessary tokens from both buckets.
        """
        with self._lock:
            self._update_buckets()

            if (
                self.request_bucket.tokens >= 1
                and self.token_bucket.tokens >= token_count
            ):
                self.request_bucket.tokens -= 1
                self.token_bucket.tokens -= token_count

                # Update stats
                self._stats["total_requests"] += 1
                self._stats["total_tokens"] += token_count
                return True

            # Calculate wait time based on token refill rates
            wait_for_request = (
                0
                if self.request_bucket.tokens >= 1
                else (1 - self.request_bucket.tokens) / self.requests_per_second
            )
            wait_for_tokens = (
                0
                if self.token_bucket.tokens >= token_count
                else (token_count - self.token_bucket.tokens) / self.tokens_per_second
            )

            # Return the longer wait time needed
            self._next_check_time = max(wait_for_request, wait_for_tokens)
            return False

    def _count_tokens(self, prompt: Optional[str] = None) -> int:
        """
        Count tokens in the prompt if available.

        Falls back to historical average if prompt is None.
        If there are no historical data, it default estimate of 100 tokens.
        """
        if prompt is None:
            # Use historical average or default
            recent_sizes = [size for _, size in self._stats["request_sizes"][-100:]]
            return int(sum(recent_sizes) / len(recent_sizes)) if recent_sizes else 100

        try:
            count = self.tokenizer(prompt)
            self._stats["request_sizes"].append((time.monotonic(), count))
            logger.debug(f"Returning Token count: {count}")
            # print(f"Token count: {count}")
            return count
        except Exception:
            logger.error(f"Error counting tokens: {e}")
            return 100  # Conservative fallback

    async def _aacquire(
        self, prompt: Optional[str] = None, *, blocking: bool = True
    ) -> bool:
        """
        Async acquire that handles token counting automatically.

        Args:
            prompt: Optional text to count tokens from
            blocking: Whether to wait for capacity

        Returns:
            True if capacity was acquired
        """
        token_count = self._count_tokens(prompt)

        if not blocking:
            return self._can_consume(token_count)

        while not self._can_consume(token_count):
            # Use the calculated wait time instead of fixed interval
            wait_time = min(self._next_check_time, self.check_every_n_seconds)
            await asyncio.sleep(wait_time)
        return True

    def _acquire(self, prompt: Optional[str] = None, *, blocking: bool = True) -> bool:
        """
        Synchronous version of acquire.

        Args:
            prompt: Optional text to count tokens from
            blocking: Whether to wait for capacity

        Returns:
            True if capacity was acquired
        """
        token_count = self._count_tokens(prompt)

        if self._can_consume(token_count):
            return True

        if not blocking:
            return False

        while not self._can_consume(token_count):
            # Use the calculated wait time instead of fixed interval
            wait_time = min(self._next_check_time, self.check_every_n_seconds)
            time.sleep(wait_time)

        return True

    async def aacquire(
        self,
        prompt: Optional[str] = None,
        *,
        blocking: bool = True,
        run_id: Optional[UUID] = None,
    ) -> bool:
        """
        Async Get prompt from either direct arg or callback.

        Args:
            prompt: Direct prompt text (if available)
            blocking: Whether to wait for capacity
            run_id: The LangChain run ID to look up prompt
        """
        if prompt is None and self.callback and run_id:
            prompt = self.callback.get_prompt(run_id)
        return await self._aacquire(prompt=prompt, blocking=blocking)

    def acquire(
        self,
        prompt: Optional[str] = None,
        *,
        blocking: bool = True,
        run_id: Optional[UUID] = None,
    ) -> bool:
        """
        Get prompt from either direct arg or callback.

        Args:
            prompt: Direct prompt text (if available)
            blocking: Whether to wait for capacity
            run_id: The LangChain run ID to look up prompt
        """

        logger.debug(f"Rate limiter acquire called with run_id: {run_id}")
        if prompt is None and self.callback and run_id:
            prompt = self.callback.get_prompt(run_id)

        return self._acquire(prompt=prompt, blocking=blocking)

    def register_callback(self, callback: "RateLimitCallback") -> None:
        """Register the callback that will provide prompts."""
        self.callback = callback

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics about rate limiter state.

        Returns detailed statistics about usage patterns and current capacity.
        """
        with self._lock:
            self._update_buckets()

            metrics = {
                "available_requests": self.request_bucket.tokens,
                "available_tokens": self.token_bucket.tokens,
                "requests_per_minute": self.requests_per_second * 60,
                "tokens_per_minute": self.tokens_per_second * 60,
                "total_requests_processed": self._stats["total_requests"],
                "total_tokens_processed": self._stats["total_tokens"],
            }

            return metrics


class RateLimitWrapper(BaseRateLimiter):
    """Wrapper to bridge LangChain's simple acquire interface with our SmartRateLimiter."""

    def __init__(self, smart_limiter: SmartRateLimiter):
        self.smart_limiter = smart_limiter
        self._thread_local = threading.local()

    def _get_current_run_id(self) -> Optional[UUID]:
        return getattr(self._thread_local, "current_run_id", None)

    def set_current_run_id(self, run_id: UUID) -> None:
        self._thread_local.current_run_id = run_id

    def acquire(self, *, blocking: bool = True) -> bool:
        """Bridge method that adds run_id to the acquire call."""
        run_id = self._get_current_run_id()
        return self.smart_limiter.acquire(blocking=blocking, run_id=run_id)

    async def aacquire(self, *, blocking: bool = True) -> bool:
        """Async version of acquire."""
        run_id = self._get_current_run_id()
        return await self.smart_limiter.aacquire(blocking=blocking, run_id=run_id)


class RateLimitCallback(BaseCallbackHandler):
    """Callback handler that manages prompt tracking for rate limiting."""

    def __init__(self, wrapper: RateLimitWrapper):
        self.wrapper = wrapper
        self._thread_local = threading.local()

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Store prompt and set current run_id."""
        # print(f"\non_llm_start called with run_id: {run_id}")
        # print(f"Prompts received: {prompts}")
        if prompts:
            if not hasattr(self._thread_local, "prompts"):
                self._thread_local.prompts = {}
            self._thread_local.prompts[run_id] = prompts[0]
            # Set the current run_id in the wrapper
            self.wrapper.set_current_run_id(run_id)

    def get_prompt(self, run_id: UUID) -> Optional[str]:
        """Get prompt for current run_id and remove it from storage."""
        if not hasattr(self._thread_local, "prompts"):
            return None
        return self._thread_local.prompts.pop(run_id, None)


class LangchainTokenAwareRateLimiter(BaseRateLimiter):
    """
    Token-aware rate limiter for LangChain that manages both request and token rate limits.

    Example Usage:
    -------------
    ```python
    from utils.llm_utils import GPT4Tokenizer

    rate_limiter = LangchainTokenAwareRateLimiter(
        requests_per_minute=60,
        tokens_per_minute=40_000,
        tokenizer=GPT4Tokenizer(),
    )

    chat = ChatVertexAI(
        model="gemini-1.5-flash-001",
        rate_limiter=rate_limiter,
        callbacks=rate_limiter.callbacks,
    )
    ```
    """

    def __init__(
        self,
        requests_per_minute: float,
        tokens_per_minute: float,
        tokenizer: Tokenizer,
        max_request_burst: Optional[float] = None,
        max_token_burst: Optional[float] = None,
        check_every_n_seconds: float = 0.1,
    ):
        """
        Initialize the rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            tokens_per_minute: Maximum tokens per minute
            tokenizer: Callable that counts tokens in text
            max_request_burst: Maximum burst size for requests
            max_token_burst: Maximum burst size for tokens
            check_every_n_seconds: Frequency to check if we can consume tokens
        """
        # Create internal components
        self._smart_limiter = SmartRateLimiter(
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            tokenizer=tokenizer,
            max_request_burst=max_request_burst,
            max_token_burst=max_token_burst,
            check_every_n_seconds=check_every_n_seconds,
        )
        self._wrapper = RateLimitWrapper(self._smart_limiter)
        self._callback = RateLimitCallback(self._wrapper)

        # Register callback with smart limiter
        self._smart_limiter.register_callback(self._callback)

        # Store for external access
        self.callbacks = [self._callback]

    def acquire(self, *, blocking: bool = True) -> bool:
        """Acquire rate limit permission."""
        return self._wrapper.acquire(blocking=blocking)

    async def aacquire(self, *, blocking: bool = True) -> bool:
        """Async acquire rate limit permission."""
        return await self._wrapper.aacquire(blocking=blocking)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics about rate limiter state."""
        return self._smart_limiter.get_metrics()
