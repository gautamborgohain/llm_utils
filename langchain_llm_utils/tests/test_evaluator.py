import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_llm_utils.evaluator import (
    ResponseLengthScore,
    CoherenceScore,
    Evaluator,
    EvaluatorConfig,
    BasicEvaluationSuite,
    ModelProvider,
    CoherenceResponse,
)
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field


class TestResponse(BaseModel):
    text: str = Field(default_factory=lambda: "test")


@pytest.fixture
def mock_llm():
    """Fixture to mock the LLM instance in CoherenceScore."""
    with patch("langchain_llm_utils.evaluator.LLM") as mock:
        mock_instance = MagicMock()
        # Set up sync mock
        mock_instance.generate.return_value = CoherenceResponse(coherence=0.8)
        # Set up async mock
        async_mock = AsyncMock()
        async_mock.return_value = CoherenceResponse(coherence=0.8)
        mock_instance.agenerate = async_mock
        mock.return_value = mock_instance
        yield mock_instance


def test_response_length_score():
    """Test the ResponseLengthScore evaluator with different input types."""
    scorer = ResponseLengthScore()

    # Test with string input
    assert scorer.evaluate("prompt", "test response") == 13

    # Test with AIMessage input
    message = AIMessage(content="test response")
    assert scorer.evaluate("prompt", message) == 13

    # Test with BaseModel input
    model = TestResponse(text="test response")
    assert scorer.evaluate("prompt", model) == 24  # Length of JSON string


@pytest.mark.asyncio
async def test_response_length_score_async():
    """Test the async version of ResponseLengthScore."""
    scorer = ResponseLengthScore()
    score = await scorer.aevaluate("prompt", "test response")
    assert score == 13


def test_coherence_score(mock_llm):
    """Test the CoherenceScore evaluator."""
    scorer = CoherenceScore(llm_provider=ModelProvider.OPENAI, llm_name="gpt-4")

    score = scorer.evaluate("test prompt", "test response")
    assert score == 0.8

    # Verify the provider was called
    mock_llm.generate.assert_called_once()


@pytest.mark.asyncio
async def test_coherence_score_async(mock_llm):
    """Test the async version of CoherenceScore."""
    scorer = CoherenceScore(llm_provider=ModelProvider.OPENAI, llm_name="gpt-4")

    score = await scorer.aevaluate("test prompt", "test response")
    assert score == 0.8

    mock_llm.agenerate.assert_called_once()


def test_evaluator_config():
    """Test EvaluatorConfig configuration."""
    config = EvaluatorConfig()

    # Add heuristic scorer
    config.add_heuristic(ResponseLengthScore())
    assert len(config.scores) == 1

    # Add LLM judge scorer
    coherence_score = CoherenceScore(
        llm_provider=ModelProvider.OPENAI, llm_name="gpt-4"
    )
    config.add_llm_judge(coherence_score)
    assert len(config.scores) == 2


def test_evaluator(mock_llm):
    """Test the main Evaluator class."""
    config = EvaluatorConfig()
    config.add_heuristic(ResponseLengthScore())
    config.add_llm_judge(
        CoherenceScore(llm_provider=ModelProvider.OPENAI, llm_name="gpt-4")
    )

    evaluator = Evaluator(config)
    results = evaluator.evaluate("test prompt", "test response")

    assert len(results) == 2
    assert results[0].score == 13  # Length score
    assert results[1].score == 0.8  # Coherence score


@pytest.mark.asyncio
async def test_evaluator_async(mock_llm):
    """Test the async version of Evaluator."""
    config = EvaluatorConfig()
    config.add_heuristic(ResponseLengthScore())
    config.add_llm_judge(
        CoherenceScore(llm_provider=ModelProvider.OPENAI, llm_name="gpt-4")
    )

    evaluator = Evaluator(config)
    results = await evaluator.aevaluate("test prompt", "test response")

    assert len(results) == 2
    assert results[0].score == 13  # Length score
    assert results[1].score == 0.8  # Coherence score


def test_basic_evaluation_suite(mock_llm):
    """Test the BasicEvaluationSuite class."""
    suite = BasicEvaluationSuite()
    results = suite.evaluate("test prompt", "test response")

    assert len(results) == 2
    assert results[0].score == 13  # Length score
    assert results[1].score == 0.8  # Coherence score


@pytest.mark.asyncio
async def test_basic_evaluation_suite_async(mock_llm):
    """Test the async version of BasicEvaluationSuite."""
    suite = BasicEvaluationSuite()
    results = await suite.aevaluate("test prompt", "test response")

    assert len(results) == 2
    assert results[0].score == 13  # Length score
    assert results[1].score == 0.8  # Coherence score


def test_evaluator_error_handling(mock_llm):
    """Test error handling in Evaluator."""
    config = EvaluatorConfig()

    # Create a scorer that raises an exception
    faulty_scorer = ResponseLengthScore()
    faulty_scorer.evaluate = MagicMock(side_effect=Exception("Test error"))

    config.add_heuristic(faulty_scorer)
    config.add_heuristic(ResponseLengthScore())

    evaluator = Evaluator(config)
    results = evaluator.evaluate("test prompt", "test response")

    # Should still get results from the working scorer
    assert len(results) == 1
    assert results[0].score == 13


@pytest.mark.asyncio
async def test_evaluator_error_handling_async(mock_llm):
    """Test error handling in async Evaluator."""
    config = EvaluatorConfig()

    # Create a scorer that raises an exception
    faulty_scorer = ResponseLengthScore()
    faulty_scorer.aevaluate = MagicMock(side_effect=Exception("Test error"))

    config.add_heuristic(faulty_scorer)
    config.add_heuristic(ResponseLengthScore())

    evaluator = Evaluator(config)
    results = await evaluator.aevaluate("test prompt", "test response")

    # Should still get results from the working scorer
    assert len(results) == 1
    assert results[0].score == 13
