import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from llm_utils.rate_limiter import SmartRateLimiter, GPT4Tokenizer
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from concurrent.futures import ThreadPoolExecutor


@pytest.fixture
def mock_tokenizer():
    """Fixture to mock the tokenizer."""
    tokenizer = Mock()
    tokenizer.return_value = 10  # Default token count for testing
    return tokenizer


@pytest.fixture
def rate_limiter(mock_tokenizer):
    """Fixture to create a rate limiter with mocked tokenizer."""
    return SmartRateLimiter(
        requests_per_minute=60,
        tokens_per_minute=4000,
        tokenizer=mock_tokenizer,
        check_every_n_seconds=0.1,
    )


@pytest.mark.asyncio
async def test_aacquire_immediate_capacity(rate_limiter):
    """Test async acquire when capacity is immediately available."""
    try:
        async with asyncio.timeout(1):  # Add timeout
            result = await rate_limiter.aacquire("test prompt")
            assert result is True
    finally:
        await rate_limiter.shutdown()


def test_acquire_sync(rate_limiter):
    """Test synchronous acquire."""
    result = rate_limiter.acquire("test prompt")
    assert result is True


def test_acquire_sync_non_blocking(rate_limiter):
    """Test synchronous acquire with non-blocking."""
    rate_limiter.request_bucket.available_tokens = 0
    result = rate_limiter.acquire("test prompt", blocking=False)
    assert result is False


def test_metrics(rate_limiter):
    """Test metrics collection and reporting."""
    rate_limiter.acquire("test1")
    rate_limiter.acquire("test2")
    metrics = rate_limiter.get_metrics()
    assert "available_requests" in metrics
    assert metrics["total_requests_processed"] == 2


def test_token_counting(mock_tokenizer):
    """Test token counting functionality."""
    rate_limiter = SmartRateLimiter(
        requests_per_minute=60, tokens_per_minute=4000, tokenizer=mock_tokenizer
    )

    token_count = rate_limiter._count_tokens("test prompt")
    assert token_count == 10

    rate_limiter._stats["request_sizes"] = []
    token_count = rate_limiter._count_tokens(None)
    assert token_count == 100


def test_priority_calculation(rate_limiter):
    """Test request priority calculation."""
    # Small request with long wait time
    small_priority = rate_limiter._calculate_priority(token_count=100, wait_time=30)

    # Large request with short wait time
    large_priority = rate_limiter._calculate_priority(token_count=1000, wait_time=5)

    # Small request with long wait should have higher priority
    assert small_priority > large_priority


def test_langchain_integration():
    """Test integration with LangChain."""
    tokenizer = Mock()
    tokenizer.return_value = 10

    rate_limiter = SmartRateLimiter(
        requests_per_minute=60,
        tokens_per_minute=4000,
        tokenizer=tokenizer,
        check_every_n_seconds=0.1,
    )

    # Create LangChain chat model with our rate limiter
    chat = ChatOpenAI(
        model_name="gpt-4",
        rate_limiter=rate_limiter,
        openai_api_key="fake-key",
    )

    def mock_generate(*args, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Test response"))]
        )

    try:
        # Patch _generate to let rate limiter execute
        with patch.object(
            chat, "_generate", side_effect=mock_generate
        ) as mock_generate:
            # Process multiple requests
            prompts = ["Test prompt"] * 10
            with ThreadPoolExecutor(max_workers=3) as executor:
                results = list(executor.map(chat.invoke, prompts))
                assert all(isinstance(result.content, str) for result in results)

            # Verify results through rate limiter metrics
            metrics = rate_limiter.get_metrics()
            assert (
                metrics["total_requests_processed"] == 10
            ), "Should have processed 10 requests"
            assert (
                metrics["total_tokens_processed"] == 1000
            ), "Should have processed 1000 tokens"
            assert (
                mock_generate.call_count == 10
            ), "Should have called generate 10 times"

    finally:
        rate_limiter.shutdown()
