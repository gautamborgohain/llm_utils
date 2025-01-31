import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from langchain_llm_utils.rate_limiter import (
    SmartRateLimiter,
    GPT4Tokenizer,
    LangchainTokenAwareRateLimiter,
)
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

    async with asyncio.timeout(1):  # Add timeout
        result = await rate_limiter.aacquire("test prompt")
        assert result is True


def test_acquire_sync(rate_limiter):
    """Test synchronous acquire."""
    result = rate_limiter.acquire("test prompt")
    assert result is True


def test_acquire_sync_non_blocking(rate_limiter):
    """Test synchronous acquire with non-blocking."""
    rate_limiter.request_bucket.tokens = 0
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


def test_langchain_integration():
    """Test integration with LangChain."""
    tokenizer = Mock()
    tokenizer.return_value = 10

    rate_limiter = SmartRateLimiter(
        requests_per_minute=60,
        tokens_per_minute=4000,
        tokenizer=tokenizer,
        max_request_burst=3,
        max_token_burst=100,
        check_every_n_seconds=0.1,
    )

    # Create LangChain chat model with our rate limiter
    chat = ChatOpenAI(
        model_name="gpt-4",
        rate_limiter=rate_limiter,
        openai_api_key="fake-key",
    )

    def mock_generate(*args, **kwargs):
        time.sleep(0.1)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Test response"))]
        )

    # Patch _generate to let rate limiter execute
    with patch.object(chat, "_generate", side_effect=mock_generate) as mock_generate:
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
        assert mock_generate.call_count == 10, "Should have called generate 10 times"


def test_langchain_integration_w_prompt_callback():
    """Test integration with LangChain."""
    tokenizer = Mock()
    tokenizer.return_value = 10

    rate_limiter = LangchainTokenAwareRateLimiter(
        requests_per_minute=60,
        tokens_per_minute=4000,
        tokenizer=tokenizer,
        max_request_burst=3,
        max_token_burst=100,
        check_every_n_seconds=0.1,
    )

    # Create LangChain chat model with our rate limiter
    chat = ChatOpenAI(
        model_name="gpt-4",
        rate_limiter=rate_limiter,
        openai_api_key="fake-key",
        callbacks=rate_limiter.callbacks,
    )

    def mock_generate(*args, **kwargs):
        time.sleep(0.1)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Test response"))]
        )

    # Patch _generate to let rate limiter execute
    with patch.object(chat, "_generate", side_effect=mock_generate) as mock_generate:
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
            metrics["total_tokens_processed"] == 100
        ), "Should have processed 100 tokens"
        assert mock_generate.call_count == 10, "Should have called generate 10 times"


@pytest.mark.skip(
    reason="Async mode on ratelimiter is not working as expected. TODO: fix"
)
@pytest.mark.asyncio
async def test_langchain_integration_w_prompt_callback_async():
    tokenizer = Mock()
    tokenizer.return_value = 10

    rate_limiter = LangchainTokenAwareRateLimiter(
        requests_per_minute=60,
        tokens_per_minute=4000,
        tokenizer=tokenizer,
        max_request_burst=3,
        max_token_burst=100,
        check_every_n_seconds=0.1,
    )
    chat = ChatOpenAI(
        model_name="gpt-4",
        rate_limiter=rate_limiter,
        openai_api_key="fake-key",
        callbacks=rate_limiter.callbacks,
    )

    def mock_generate(*args, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Test response"))]
        )

    with patch.object(chat, "_agenerate", side_effect=mock_generate) as mock_generate:
        # Process multiple requests concurrently using asyncio
        prompts = ["Test prompt"] * 10
        tasks = [chat.ainvoke(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        assert all(isinstance(result.content, str) for result in results)

        metrics = rate_limiter.get_metrics()
        assert (
            metrics["total_requests_processed"] == 10
        ), "Should have processed 10 requests"
        assert (
            metrics["total_tokens_processed"] == 100
        ), "Should have processed 100 tokens"
        assert mock_generate.call_count == 10, "Should have called generate 10 times"


@pytest.mark.skip(
    reason="This test simulates a large batch of requests, and thus take >1 minute to run. Recommend adhoc testing."
)
def test_langchain_integration_large_batch():
    """Test integration with LangChain for a large batch of requests."""
    tokenizer = Mock()
    tokenizer.return_value = 10

    rate_limiter = LangchainTokenAwareRateLimiter(
        requests_per_minute=60,  # 1 request/second
        tokens_per_minute=4000,  # ~67 tokens/second
        tokenizer=tokenizer,
        max_request_burst=3,  # Only 3 concurrent requests
        max_token_burst=100,  # Enough for concurrent requests
        check_every_n_seconds=0.1,
    )

    chat = ChatOpenAI(
        model_name="gpt-4",
        rate_limiter=rate_limiter,
        openai_api_key="fake-key",
        callbacks=rate_limiter.callbacks,
    )

    def mock_generate(*args, **kwargs):
        # time.sleep(0.1)  # Simulate some processing time
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Test response"))]
        )

    with patch.object(chat, "_generate", side_effect=mock_generate) as mock_generate:
        start_time = time.time()

        # Process 100 requests
        prompts = ["Test prompt"] * 100
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(chat.invoke, prompts))
            assert all(isinstance(result.content, str) for result in results)

        end_time = time.time()
        processing_time = end_time - start_time

        # Verify metrics
        metrics = rate_limiter.get_metrics()
        print(
            f"Processed {100/processing_time:.2f} requests per second; Time taken: {processing_time:.2f} seconds"
        )

        print(metrics)
        assert metrics["total_requests_processed"] == 100, "Should process all requests"
        assert metrics["total_tokens_processed"] == 1000, "Should process all tokens"
        assert mock_generate.call_count == 100, "Should make all calls"

        # Should take roughly 100 seconds (1 request/second)
        assert (
            95 <= processing_time <= 180
        ), f"Processing time was {processing_time} seconds"
