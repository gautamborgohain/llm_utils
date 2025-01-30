from pydantic import BaseModel
from typing import TypeVar
import logging
from logging import getLogger, StreamHandler
import os
import sys
import hashlib
from uuid import UUID


def get_logger(name: str = None):
    logger = getLogger(name or __name__)
    logger.setLevel(os.getenv("LOG_LEVEL", "DEBUG"))

    # allow stdout to be logged
    logger.addHandler(logging.StreamHandler(sys.stdout))

    # log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.handlers[0].setFormatter(formatter)
    return logger


logger = get_logger(__name__)

T = TypeVar("T")  # Generic type for input
R = TypeVar("R")  # Generic type for result
BaseModelType = TypeVar("BaseModelType", bound=BaseModel)


def decider(uuid: UUID, sample_rate: float, hash_salt: str = "") -> bool:
    """
    Deterministically decide whether to sample a UUID based on sample rate.

    Args:
        uuid: UUID to check
        sample_rate: Float between 0 and 1 representing sampling probability
        hash_salt: Optional salt string to mix into hash

    Returns:
        bool: True if UUID should be sampled based on rate, False otherwise
    """
    if sample_rate <= 0:
        return False
    if sample_rate >= 1:
        return True

    # Create hash of UUID + salt
    hash_input = f"{uuid}{hash_salt}".encode("utf-8")
    hash_value = hashlib.md5(hash_input).hexdigest()

    # Convert first 4 bytes of hash to int and normalize to 0-1
    hash_int = int(hash_value[:8], 16)
    normalized = hash_int / 0xFFFFFFFF

    return normalized < sample_rate
