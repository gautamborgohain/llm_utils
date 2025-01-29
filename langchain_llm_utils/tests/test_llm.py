import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pydantic import BaseModel, Field
from typing import List, Callable
from langchain_llm_utils.llm import LLM, ModelProvider


class JokeResponse(BaseModel):
    setup: str = Field(default="What do you call a fish with no eyes?")
    punchline: str = Field(default="A fsh!")


@pytest.fixture
def mock_llm_provider():
    """Fixture to mock the LLM provider."""
    with patch(
        "langchain_llm_utils.llm.LLMFactory.create_provider"
    ) as mock_create_provider:
        mock_provider = MagicMock()
        # Set up sync mock
        mock_provider.invoke.return_value = "Mocked response"

        # Set up async mock
        async_mock = AsyncMock()
        async_mock.return_value = "Mocked async response"
        mock_provider.ainvoke = async_mock

        mock_create_provider.return_value = mock_provider
        yield mock_provider


@pytest.fixture
def llm_instance(mock_llm_provider):
    """Fixture to create an LLM instance for testing."""
    return LLM(model_provider=ModelProvider.OPENAI, model_name="gpt-4o")


@pytest.fixture
def llm_instance_with_response_model(mock_llm_provider):
    """Fixture to create an LLM instance with a response model for testing structured output."""
    return LLM(
        model_provider=ModelProvider.OPENAI,
        model_name="gpt-4o",
        response_model=JokeResponse,
    )


def test_invoke(mock_llm_provider, llm_instance):
    """Test the invoke method."""
    # Mock the _llm.invoke method to return a specific response
    mock_llm_provider.invoke.return_value = "Mocked response"

    response = llm_instance.invoke("What is the capital of France?")
    assert response == "Mocked response"
    mock_llm_provider.invoke.assert_called_once_with("What is the capital of France?")


def test_invoke_connection_error(mock_llm_provider, llm_instance):
    """Test that connection errors are re-raised in invoke method."""
    # Mock the provider to raise a ConnectionRefusedError
    mock_llm_provider.invoke.side_effect = ConnectionRefusedError("Connection refused")

    with pytest.raises(ConnectionRefusedError) as exc_info:
        llm_instance.invoke("Test prompt")
    assert "Failed to connect to LLM" in str(exc_info.value)


def test_invoke_other_exception(mock_llm_provider, llm_instance):
    """Test that non-connection errors return None in invoke method."""
    # Mock the provider to raise a generic exception
    mock_llm_provider.invoke.side_effect = Exception("Some other error")

    response = llm_instance.invoke("Test prompt")
    assert response is None


def test_generate(mock_llm_provider, llm_instance):
    """Test the generate method."""
    # Mock the _llm.invoke method to return a specific response
    mock_llm_provider.invoke.return_value = "Mocked response"

    response = llm_instance.generate("Tell me a joke")
    assert response == "Mocked response"
    mock_llm_provider.invoke.assert_called_once_with("Tell me a joke")


def test_generate_structured_output(
    llm_instance_with_response_model, mock_llm_provider
):
    """Test the generate method with structured output."""
    # Mock the _llm.invoke method to return a structured response
    mock_response = JokeResponse()
    mock_llm_provider.invoke.return_value = mock_response

    response = llm_instance_with_response_model.generate("Tell me a joke")

    assert isinstance(response, JokeResponse)
    assert response.setup == mock_response.setup
    assert response.punchline == mock_response.punchline


def test_generate_structured_output_when_no_response_from_llm(
    llm_instance_with_response_model, mock_llm_provider
):
    """Test the generate method with structured output."""
    # Mock the _llm.invoke method to return None (failed LLM call)
    mock_llm_provider.invoke.return_value = None

    response = llm_instance_with_response_model.generate("Tell me a joke")
    assert isinstance(
        response, JokeResponse
    )  # Make sure response is of type JokeResponse (default response model)


def test_generate_with_template(mock_llm_provider, llm_instance):
    """Test the generate method with a template and input variables."""
    # Mock the _llm.invoke method to return a specific response
    mock_llm_provider.invoke.return_value = "Mocked response"

    response = llm_instance.generate(
        template="Tell me a {type} joke about {topic}",
        input_variables={"type": "dad", "topic": "programming"},
    )
    assert (
        response == "Mocked response"
    )  # Check if the response matches the mocked response
    mock_llm_provider.invoke.assert_called_once_with(
        "Tell me a dad joke about programming"
    )


@pytest.mark.asyncio
async def test_ainvoke(mock_llm_provider, llm_instance):
    """Test the async invoke method."""
    # Mock the _llm.ainvoke method to return a specific response
    mock_llm_provider.ainvoke.return_value = "Mocked async response"

    response = await llm_instance.ainvoke("What is the capital of France?")
    assert response == "Mocked async response"
    mock_llm_provider.ainvoke.assert_called_once_with("What is the capital of France?")


@pytest.mark.asyncio
async def test_ainvoke_connection_error(mock_llm_provider, llm_instance):
    """Test that connection errors are re-raised in async invoke method."""
    # Mock the provider to raise a ConnectionRefusedError
    mock_llm_provider.ainvoke.side_effect = ConnectionRefusedError("Connection refused")

    with pytest.raises(ConnectionRefusedError) as exc_info:
        await llm_instance.ainvoke("Test prompt")
    assert "Failed to connect to LLM" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ainvoke_other_exception(mock_llm_provider, llm_instance):
    """Test that non-connection errors return None in async invoke method."""
    # Mock the provider to raise a generic exception
    mock_llm_provider.ainvoke.side_effect = Exception("Some other error")

    response = await llm_instance.ainvoke("Test prompt")
    assert response is None


@pytest.mark.asyncio
async def test_agenerate(mock_llm_provider, llm_instance):
    """Test the async generate method."""
    # Mock the _llm.ainvoke method to return a specific response
    mock_llm_provider.ainvoke.return_value = "Mocked async response"

    response = await llm_instance.agenerate("Tell me a joke")
    assert response == "Mocked async response"
    mock_llm_provider.ainvoke.assert_called_once_with("Tell me a joke")


@pytest.mark.asyncio
async def test_agenerate_structured_output(
    llm_instance_with_response_model, mock_llm_provider
):
    """Test the async generate method with structured output."""
    # Mock the _llm.ainvoke method to return a structured response
    mock_response = JokeResponse()
    mock_llm_provider.ainvoke.return_value = mock_response

    response = await llm_instance_with_response_model.agenerate("Tell me a joke")

    assert isinstance(response, JokeResponse)
    assert response.setup == mock_response.setup
    assert response.punchline == mock_response.punchline


@pytest.mark.asyncio
async def test_agenerate_structured_output_when_no_response_from_llm(
    llm_instance_with_response_model, mock_llm_provider
):
    """Test the async generate method with structured output when LLM returns None."""
    # Mock the _llm.ainvoke method to return None (failed LLM call)
    mock_llm_provider.ainvoke.return_value = None

    response = await llm_instance_with_response_model.agenerate("Tell me a joke")
    assert isinstance(
        response, JokeResponse
    )  # Make sure response is of type JokeResponse (default response model)


@pytest.mark.asyncio
async def test_agenerate_with_template(mock_llm_provider, llm_instance):
    """Test the async generate method with a template and input variables."""
    # Mock the _llm.ainvoke method to return a specific response
    mock_llm_provider.ainvoke.return_value = "Mocked async response"

    response = await llm_instance.agenerate(
        template="Tell me a {type} joke about {topic}",
        input_variables={"type": "dad", "topic": "programming"},
    )
    assert response == "Mocked async response"
    mock_llm_provider.ainvoke.assert_called_once_with(
        "Tell me a dad joke about programming"
    )
