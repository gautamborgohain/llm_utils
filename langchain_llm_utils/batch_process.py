from typing import List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain_llm_utils.common import T, R
from langchain_llm_utils.common import get_logger

logger = get_logger("BatchProcessor")


class WorkloadType(Enum):
    CPU_BOUND = "cpu_bound"
    IO_BOUND = "io_bound"
    MIXED = "mixed"


@dataclass
class BatchProcessConfig:
    workload_type: WorkloadType
    cpu_allocation: float = 0.75
    min_workers: int = 1
    max_workers: Optional[int] = None
    description: str = "Processing"
    unit: str = "item"


def batch_process_with_progress(
    items: List[T],
    process_func: Callable[[T], R],
    config: BatchProcessConfig,
) -> List[R]:
    """
    Generic utility for batch processing items with real-time progress tracking.

    Args:
        items: List of items to process
        process_func: Function to process each item
        config: BatchProcessConfig object containing processing parameters
                config.workload_type :
                    If config.workload_type is CPU_BOUND, then the number of workers will be
                    the number of CPUs minus 1.
                    If config.workload_type is IO_BOUND, then the number of workers will be
                    twice the number of CPUs.
                    If config.workload_type is MIXED, then the number of workers will be
                    the number of CPUs times the cpu_allocation.
                config.max_workers :
                    If config.max_workers is not None, then the number of workers will be
                    the minimum of the number of workers calculated above and config.max_workers.

    Returns:
        List of processed results
    """
    cpu_count = multiprocessing.cpu_count()

    # Calculate optimal workers
    if config.workload_type == WorkloadType.CPU_BOUND:
        optimal_workers = max(config.min_workers, cpu_count - 1)
    elif config.workload_type == WorkloadType.IO_BOUND:
        optimal_workers = cpu_count * 2
    else:  # MIXED
        optimal_workers = max(
            config.min_workers, int(cpu_count * config.cpu_allocation)
        )

    if config.max_workers:
        optimal_workers = min(optimal_workers, config.max_workers)

    # Initialize results list with None values to maintain order
    results = [None] * len(items)

    with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
        # Submit all tasks and store futures with their indices
        future_to_index = {
            executor.submit(process_func, item): idx for idx, item in enumerate(items)
        }

        # Create progress bar
        with tqdm(total=len(items), desc=config.description, unit=config.unit) as pbar:
            # Process completed futures as they finish
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    results[idx] = e
                pbar.update(1)

    # Check for exceptions
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            raise result
    return results
