from pydantic import BaseModel
from typing import TypeVar
from logging import getLogger

logger = getLogger(__name__)

T = TypeVar("T")  # Generic type for input
R = TypeVar("R")  # Generic type for result
BaseModelType = TypeVar("BaseModelType", bound=BaseModel)
