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
    EvaluatorCallback,
)
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from uuid import UUID
import time
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.outputs import ChatResult, ChatGeneration
from concurrent.futures import ThreadPoolExecutor


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


@pytest.mark.parametrize(
    "sample_rate,expected_results",
    [
        (1.0, 2),  # Full sampling - should always evaluate
        (0.0, 0),  # No sampling - should never evaluate
    ],
)
def test_basic_evaluation_suite_sampling(mock_llm, sample_rate, expected_results):
    """Test that BasicEvaluationSuite respects sampling rates."""
    suite = BasicEvaluationSuite(sample_rate=sample_rate)
    results = suite.evaluate("test prompt", "test response")
    assert len(results) == expected_results


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sample_rate,expected_results",
    [
        (1.0, 2),  # Full sampling - should always evaluate
        (0.0, 0),  # No sampling - should never evaluate
    ],
)
async def test_basic_evaluation_suite_sampling_async(
    mock_llm, sample_rate, expected_results
):
    """Test that async BasicEvaluationSuite respects sampling rates."""
    suite = BasicEvaluationSuite(sample_rate=sample_rate)
    results = await suite.aevaluate("test prompt", "test response")
    assert len(results) == expected_results


def test_evaluator_sampling_with_run_id(mock_llm):
    """Test that Evaluator sampling works consistently with run_ids."""
    # Create a fixed UUID for testing
    test_uuid = UUID("12345678-1234-5678-1234-567812345678")

    # Create evaluator with 50% sampling rate
    suite = BasicEvaluationSuite(sample_rate=0.5)

    # First evaluation with the UUID should determine if subsequent
    # evaluations with the same UUID are processed
    first_results = suite.evaluate("test prompt", "test response", run_id=test_uuid)
    expected_length = len(first_results)  # Will be either 0 or 2

    # Subsequent evaluations with same UUID should match first result
    second_results = suite.evaluate("test prompt", "test response", run_id=test_uuid)
    assert len(second_results) == expected_length


@pytest.mark.asyncio
async def test_evaluator_sampling_with_run_id_async(mock_llm):
    """Test that async Evaluator sampling works consistently with run_ids."""
    # Create a fixed UUID for testing
    test_uuid = UUID("12345678-1234-5678-1234-567812345678")

    # Create evaluator with 50% sampling rate
    suite = BasicEvaluationSuite(sample_rate=0.5)

    # First evaluation with the UUID should determine if subsequent
    # evaluations with the same UUID are processed
    first_results = await suite.aevaluate(
        "test prompt", "test response", run_id=test_uuid
    )
    expected_length = len(first_results)  # Will be either 0 or 2

    # Subsequent evaluations with same UUID should match first result
    second_results = await suite.aevaluate(
        "test prompt", "test response", run_id=test_uuid
    )
    assert len(second_results) == expected_length


def test_evaluator_langchain_integration(mock_llm):
    """Test Evaluator integration with LangChain."""

    # Create evaluator with full sampling
    suite = BasicEvaluationSuite(sample_rate=1.0)

    # Create LangChain chat model with our evaluator callback
    chat = ChatOpenAI(
        model_name="gpt-4",
        callbacks=suite.callbacks,
        openai_api_key="fake-key",
    )

    def mock_generate_result(*args, **kwargs):
        time.sleep(0.1)  # Simulate API call
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Test response"))]
        )

    # We need to mock both the chat._generate and the coherence score evaluation
    with patch.object(
        chat, "_generate", side_effect=mock_generate_result
    ) as mock_generate, patch.object(
        mock_llm, "generate", return_value=CoherenceResponse(coherence=0.8)
    ):

        # Process multiple requests
        prompts = ["Test prompt"] * 10
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(chat.invoke, prompts))
            assert all(isinstance(result.content, str) for result in results)

        # Verify all requests were evaluated (since sample_rate=1.0)
        assert mock_generate.call_count == 10

        # Verify that evaluations were performed via callback
        # We need to add a way to track evaluations in the Evaluator class
        assert (
            len(suite.evaluation_results) == 10
        )  # This would need to be added to Evaluator

        # Verify the evaluation results
        for eval_result in suite.evaluation_results:
            assert len(eval_result) == 2  # Both scorers should have run
            assert eval_result[0].score_name == "ResponseLengthScore"
            assert eval_result[1].score_name == "CoherenceScore"
            assert eval_result[1].score == 0.8  # From our mocked coherence score


def test_evaluator_langchain_integration_with_sampling(mock_llm):
    """Test Evaluator integration with LangChain with partial sampling rate."""
    # Create evaluator with 70% sampling rate
    suite = BasicEvaluationSuite(sample_rate=0.7)
    # evaluator_callback = EvaluatorCallback(suite)

    # Create LangChain chat model with our evaluator callback
    chat = ChatOpenAI(
        model_name="gpt-4",
        callbacks=suite.callbacks,
        openai_api_key="fake-key",
    )

    def mock_generate_result(*args, **kwargs):
        time.sleep(0.1)  # Simulate API call
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="Test response"))]
        )

    # We need to mock both the chat._generate and the coherence score evaluation
    with patch.object(
        chat, "_generate", side_effect=mock_generate_result
    ) as mock_generate, patch.object(
        mock_llm, "generate", return_value=CoherenceResponse(coherence=0.8)
    ):
        # Process 100 requests to get a statistically significant sample
        num_requests = 100
        prompts = ["Test prompt"] * num_requests
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(chat.invoke, prompts))
            assert all(isinstance(result.content, str) for result in results)

        # Verify all requests were processed
        assert mock_generate.call_count == num_requests

        # Verify that evaluations were performed via callback
        # The number of evaluations should be approximately 70% of requests
        num_evaluations = len(suite.evaluation_results)
        expected_evaluations = int(num_requests * 0.7)
        margin_of_error = int(num_requests * 0.1)  # Allow 10% margin of error
        print("--------------------------------")
        print(f"Number of requests made to LLM: {mock_generate.call_count}")
        print(
            f"Number of evaluations performed: {num_evaluations}, "
            f"expected evaluations: {expected_evaluations}, "
            f"with margin of error +-{margin_of_error}"
        )
        assert abs(num_evaluations - expected_evaluations) <= margin_of_error, (
            f"Expected approximately {expected_evaluations} evaluations "
            f"(±{margin_of_error}), but got {num_evaluations}"
        )

        # Verify the evaluation results structure
        for eval_result in suite.evaluation_results:
            assert len(eval_result) == 2  # Both scorers should have run
            assert eval_result[0].score_name == "ResponseLengthScore"
            assert eval_result[1].score_name == "CoherenceScore"
            assert eval_result[1].score == 0.8  # From our mocked coherence score
