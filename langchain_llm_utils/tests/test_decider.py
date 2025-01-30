import uuid
from langchain_llm_utils.common import decider


def test_decider_edge_cases():
    test_uuid = uuid.uuid4()

    # Should never sample when rate is 0
    assert not decider(test_uuid, 0.0)
    assert not decider(test_uuid, -0.1)

    # Should always sample when rate is 1 or greater
    assert decider(test_uuid, 1.0)
    assert decider(test_uuid, 1.1)


def test_decider_deterministic():
    # Same inputs should give same results
    test_uuid = uuid.uuid4()
    result1 = decider(test_uuid, 0.5)
    result2 = decider(test_uuid, 0.5)
    assert result1 == result2


def test_decider_distribution():
    # Generate many UUIDs and verify sampling rate is approximately correct
    sample_rate = 0.3
    num_trials = 10000
    sampled = 0

    for _ in range(num_trials):
        if decider(uuid.uuid4(), sample_rate):
            sampled += 1

    # Calculate actual rate
    actual_rate = sampled / num_trials

    # Allow for some statistical variance (within ±2 percentage points)
    assert (
        abs(actual_rate - sample_rate) < 0.02
    ), f"Expected rate ~{sample_rate}, got {actual_rate}"
