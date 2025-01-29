import pytest
import json
import os
from unittest.mock import patch, MagicMock
from pydantic import BaseModel, Field
from langchain_llm_utils.request_cache import RequestCache
from langchain_llm_utils.llm import LLM, ModelProvider


class JokeResponse(BaseModel):
    setup: str = Field(default="What do you call a fish with no eyes?")
    punchline: str = Field(default="A fsh!")


@pytest.fixture
def mock_cache_file(tmp_path):
    """Fixture to create a temporary cache file. If the file already exists, it will be cleared."""
    cache_file = tmp_path / "test_cache.json"
    if os.path.exists(cache_file):
        os.remove(cache_file)
    return str(cache_file)


@pytest.fixture
def request_cache(mock_cache_file):
    """Fixture to create a RequestCache instance."""
    return RequestCache(cache_file=mock_cache_file)


@pytest.fixture
def mock_llm_provider():
    """Fixture to mock the LLM provider."""
    with patch(
        "langchain_llm_utils.llm.LLMFactory.create_provider"
    ) as mock_create_provider:
        mock_provider = MagicMock()
        mock_provider.with_structured_output.return_value = mock_provider
        mock_create_provider.return_value = mock_provider
        yield mock_provider


@pytest.fixture
def llm_instance_with_response_model(mock_llm_provider, request_cache):
    """Fixture to create an LLM instance with a response model for testing structured output."""
    return LLM(
        model_provider=ModelProvider.OPENAI,
        model_name="gpt-4o",
        response_model=JokeResponse,
        cache=request_cache,
    )


def test_cache_initialization(mock_cache_file):
    """Test cache initialization with new and existing cache files."""
    # Test with new cache file
    cache = RequestCache(cache_file=mock_cache_file)
    assert os.path.exists(mock_cache_file)


def test_cache_key_generation(request_cache):
    """Test cache key generation with different inputs."""
    key1 = request_cache.get_cache_key(
        model_name="gpt-4",
        model_provider=ModelProvider.OPENAI,
        final_prompt="test prompt",
    )
    key2 = request_cache.get_cache_key(
        model_name="gpt-4",
        model_provider=ModelProvider.OPENAI,
        final_prompt="test prompt",
    )
    key3 = request_cache.get_cache_key(
        model_name="gpt-4",
        model_provider=ModelProvider.OPENAI,
        final_prompt="different prompt",
    )

    # Same inputs should generate same key
    assert key1 == key2
    # Different prompts should generate different keys
    assert key1 != key3


def test_cache_set_get(request_cache):
    """Test basic cache set and get operations."""
    # Test with string value
    request_cache.set("key1", "value1")
    assert request_cache.get("key1") == "value1"

    # Test with dict value
    dict_value = {"nested": "value"}
    request_cache.set("key2", dict_value)
    assert request_cache.get("key2") == dict_value

    # Test non-existent key
    assert request_cache.get("nonexistent") is None


def test_cache_persistence(mock_cache_file):
    """Test that cache data persists between instances."""
    cache1 = RequestCache(cache_file=mock_cache_file)
    cache1.set("key1", "value1")
    cache1.save_cache(overwrite=True)

    # Create new instance pointing to same file
    cache2 = RequestCache(cache_file=mock_cache_file)
    assert cache2.get("key1") == "value1"


def test_cache_with_invalid_structured_output(request_cache):
    """Test cache with invalid structured output data."""
    invalid_data = {"invalid": "data"}
    request_cache.set("invalid_key", invalid_data)

    # Should return default model when data doesn't match schema
    result = request_cache.get_parsed("invalid_key", JokeResponse)
    assert isinstance(result, JokeResponse)
    assert result.setup == "What do you call a fish with no eyes?"


def test_generate_structured_output(
    llm_instance_with_response_model, mock_llm_provider, request_cache
):
    """Test the generate method with structured output."""
    # Mock the _llm.invoke method to return a structured response
    request_cache.clear()
    mock_response = JokeResponse()
    mock_llm_provider.invoke.return_value = mock_response

    response = llm_instance_with_response_model.generate("Tell me a joke")

    assert isinstance(response, JokeResponse)
    assert response.setup == mock_response.setup
    assert response.punchline == mock_response.punchline


def test_cache_with_llm_integration(
    mock_llm_provider, llm_instance_with_response_model, request_cache
):
    """Test cache integration with LLM class."""
    request_cache.clear()

    mock_response = JokeResponse()
    mock_llm_provider.invoke.return_value = mock_response

    # First call should hit the LLM since cache is empty
    result1 = llm_instance_with_response_model.generate("Tell me a joke")
    assert isinstance(result1, JokeResponse)
    assert result1.setup == mock_response.setup
    assert result1.punchline == mock_response.punchline
    assert mock_llm_provider.invoke.call_count == 1

    # Second call should hit cache
    result2 = llm_instance_with_response_model.generate("Tell me a joke")
    assert isinstance(result2, JokeResponse)
    assert result2.setup == mock_response.setup
    assert result2.punchline == mock_response.punchline
    assert mock_llm_provider.invoke.call_count == 1  # Shouldn't increase


def test_cache_clear(request_cache):
    """Test cache clearing functionality."""
    request_cache.set("key1", "value1")
    request_cache.set("key2", "value2")

    request_cache.clear()

    assert request_cache.get("key1") is None
    assert request_cache.get("key2") is None
    # Check if cache is empty rather than file size
    assert request_cache.cache == {}


def test_cache_with_template_variables(
    mock_llm_provider, llm_instance_with_response_model, request_cache
):
    """Test cache behavior with template variables in LLM."""
    request_cache.clear()
    mock_response = JokeResponse()
    mock_llm_provider.invoke.return_value = mock_response

    # Test with template and variables
    template = "Tell me a {type} joke about {topic}"
    variables = {"type": "programming", "topic": "arrays"}

    result1 = llm_instance_with_response_model.generate(
        template=template, input_variables=variables
    )
    assert mock_llm_provider.invoke.call_count == 1

    # Same template and variables should hit cache
    result2 = llm_instance_with_response_model.generate(
        template=template, input_variables=variables
    )
    assert mock_llm_provider.invoke.call_count == 1
    assert result1 == result2
