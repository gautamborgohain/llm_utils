import pytest
from unittest.mock import patch, MagicMock
from langchain_llm_utils.batch_process import (
    batch_process_with_progress,
    BatchProcessConfig,
    WorkloadType,
)
from langchain_llm_utils.llm import LLM


@pytest.fixture
def mock_llm_provider():
    """Fixture to mock the LLM provider."""
    with patch(
        "langchain_llm_utils.llm.LLMFactory.create_provider"
    ) as mock_create_provider:
        mock_provider = MagicMock()
        mock_create_provider.return_value = mock_provider
        yield mock_provider


def test_batch_process_with_progress_success(mock_llm_provider):
    """Test successful batch processing."""
    # Mock the _llm.invoke method to return processed items
    mock_llm_provider.invoke.side_effect = lambda x: f"Processed {x}"
    items = ["item1", "item2", "item3"]
    config = BatchProcessConfig(workload_type=WorkloadType.CPU_BOUND)

    results = batch_process_with_progress(items, mock_llm_provider.invoke, config)

    assert results == ["Processed item1", "Processed item2", "Processed item3"]
    assert mock_llm_provider.invoke.call_count == len(items)


def test_batch_process_with_progress_error_handling(mock_llm_provider):
    """Test error handling in batch processing."""
    # Mock the _llm.invoke method to return None, empty string, or return a processed item
    mock_llm_provider.invoke.side_effect = [
        None,  # Simulate a None response for item1
        "",  # Simulate an empty string for item2
        "Processed item3",
    ]
    items = ["item1", "item2", "item3"]
    config = BatchProcessConfig(workload_type=WorkloadType.CPU_BOUND)

    results = batch_process_with_progress(items, mock_llm_provider.invoke, config)

    # Check that the results contain None for the failed item
    assert results == [None, "", "Processed item3"]
    assert mock_llm_provider.invoke.call_count == len(items)


def test_batch_process_with_progress_io_bound(mock_llm_provider):
    """Test batch processing with IO_BOUND workload type."""
    # Mock the _llm.invoke method to return processed items
    mock_llm_provider.invoke.side_effect = lambda x: f"Processed {x}"
    items = ["item1", "item2", "item3"]
    config = BatchProcessConfig(workload_type=WorkloadType.IO_BOUND)

    results = batch_process_with_progress(items, mock_llm_provider.invoke, config)

    assert results == ["Processed item1", "Processed item2", "Processed item3"]
    assert mock_llm_provider.invoke.call_count == len(items)


def test_batch_process_with_progress_mixed_workload(mock_llm_provider):
    """Test batch processing with MIXED workload type."""
    # Mock the _llm.invoke method to return processed items
    mock_llm_provider.invoke.side_effect = lambda x: f"Processed {x}"
    items = ["item1", "item2", "item3"]
    config = BatchProcessConfig(workload_type=WorkloadType.MIXED, cpu_allocation=0.5)

    results = batch_process_with_progress(items, mock_llm_provider.invoke, config)

    assert results == ["Processed item1", "Processed item2", "Processed item3"]
    assert mock_llm_provider.invoke.call_count == len(items)
