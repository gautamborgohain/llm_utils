import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel, Field
from typing import List
from llm_utils.llm import LLM, ModelProvider


class JokeResponse(BaseModel):
    setup: str = Field(default="What do you call a fish with no eyes?")
    punchline: str = Field(default="A fsh!")


@pytest.fixture
def mock_llm_provider():
    """Fixture to mock the LLM provider."""
    with patch("llm_utils.llm.LLMFactory.create_provider") as mock_create_provider:
        mock_provider = MagicMock()
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
    mock_llm_provider.invoke.return_value = {
        "setup": "What do you call a fish with no eyes?",
        "punchline": "A fsh!",
    }

    response = llm_instance_with_response_model.generate("Tell me a joke")

    assert isinstance(
        response, JokeResponse
    )  # Check if response is of type JokeResponse
    assert (
        response.setup == "What do you call a fish with no eyes?"
    )  # Ensure setup matches
    assert response.punchline == "A fsh!"  # Ensure punchline matches


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
