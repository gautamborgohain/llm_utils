from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel, Field
from mmh3 import hash_bytes
import base64
import json
import os
from langchain_llm_utils.common import get_logger, BaseModelType

logger = get_logger("RequestCache")


class RequestCache:
    """
    Cache for storing and retrieving structured responses.

    This class provides a persistent cache for storing LLM responses, with support for:
    1. Storing and retrieving Pydantic models
    2. Efficient key hashing for large prompts
    3. Persistent storage in JSON format
    4. Both raw and parsed data access

    Example usage:

    ```python
    # Basic usage
    cache = RequestCache("responses_cache.json")

    # With LLM
    from llm_utils.llm import LLM

    class TranslationResponse(BaseModel):
        translation: str
        source_language: str

    llm = LLM(
        model="gemini-1.5-flash",
        cache=cache,
        response_model=TranslationResponse
    )

    # Direct cache access
    response = cache.get_parsed("some_key", TranslationResponse)
    if response:
        print(f"Cached translation: {response.translation}")

    # Storing new response
    new_response = TranslationResponse(
        translation="Hello",
        source_language="English"
    )
    cache.set("some_key", new_response)

    # Save changes to disk
    cache.save_cache()

    # Force overwrite existing cache file
    cache.save_cache(overwrite=True)
    ```

    """

    def __init__(self, cache_file: str):
        """
        Initialize the cache.

        Args:
            cache_file: Path to the JSON file where cache will be stored
                If file doesn't exist, an empty cache will be created
                If file exists, it will be loaded

        Example:
            ```python
            # Create/load cache
            cache = RequestCache("my_cache.json")

            # Create cache in specific directory
            cache = RequestCache("data/caches/responses.json")
            ```
        """
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        try:
            with open(self.cache_file, "r") as f:
                logger.info(f"Loading cache from {self.cache_file}")
                return json.load(f)
        except FileNotFoundError:
            logger.info(f"Cache file {self.cache_file} not found, creating one ..")
            # Try to create the file and return an empty dict
            try:
                with open(self.cache_file, "w") as f:
                    return {}
            except Exception as e:
                logger.error(f"Failed to create cache file {self.cache_file}: {e}")
                return {}

    def _hash_key(self, key: str) -> str:
        """
        Generate a deterministic hash for a cache key.

        Args:
            key: The original key string

        Returns:
            A base64-encoded MurmurHash3 hash of the key
        """
        # Get bytes hash from MurmurHash3
        hash_value = hash_bytes(key.encode("utf-8"))
        # Convert to base64 for readability and storage efficiency
        return base64.urlsafe_b64encode(hash_value).decode("ascii")

    def get(self, key: str) -> Optional[Dict]:
        """
        Get raw cached value.

        Args:
            key: Cache key to lookup

        Returns:
            Raw cached dictionary or None if key not found
        """
        hashed_key = self._hash_key(key)
        # logger.debug(f"Hashed key: {hashed_key}")
        # logger.debug(f"Result: {self.cache.get(hashed_key)}")
        return self.cache.get(hashed_key)

    def get_cache_key(self, **kwargs) -> str:
        """
        Get the cache key for a given key.

        Args:
            kwargs: Keyword arguments to include in the cache key

        Returns:
            A base64-encoded MurmurHash3 hash of the key
        """
        # Create : delimited string from kwargs
        key_str = ":".join([f"{k}__{v}" for k, v in kwargs.items()])
        return self._hash_key(key_str)

    def get_parsed(
        self, key: str, model: Type[BaseModelType]
    ) -> Optional[BaseModelType]:
        """
        Get a cached value and parse it into a Pydantic model.

        Args:
            key: Cache key to lookup
            model: Pydantic model class to parse the cached data into

        Returns:
            Parsed Pydantic model instance or None if key not found
        """
        cached_result = self.get(key)
        if cached_result is None:
            return None

        def recursive_construct(cls, data):
            if isinstance(data, dict):
                for field_name, field_value in data.items():
                    field_info = cls.model_fields.get(field_name)
                    if (
                        field_info
                        and isinstance(field_info.annotation, type)
                        and issubclass(field_info.annotation, BaseModel)
                    ):
                        data[field_name] = recursive_construct(
                            field_info.annotation, field_value
                        )
                    elif (
                        field_info
                        and isinstance(field_info.annotation, type)
                        and issubclass(field_info.annotation, List)
                    ):
                        if (
                            hasattr(field_info.annotation, "__args__")
                            and isinstance(field_info.annotation.__args__[0], type)
                            and issubclass(field_info.annotation.__args__[0], BaseModel)
                        ):
                            data[field_name] = [
                                recursive_construct(
                                    field_info.annotation.__args__[0], item
                                )
                                for item in field_value
                            ]
            return cls(**data)

        try:
            return model.model_validate(cached_result)
        except Exception as e:
            logger.error(
                f"Failed to parse cached result with pydantic v2 method, trying recursive construct: {e}"
            )
            return recursive_construct(model, cached_result)

    def set(self, key: str, value: BaseModel) -> None:
        """
        Store a Pydantic model in the cache.

        Args:
            key: Cache key
            value: Pydantic model instance to store
        """
        hashed_key = self._hash_key(key)
        if isinstance(value, BaseModel):
            self.cache[hashed_key] = value.model_dump()
        else:
            self.cache[hashed_key] = value

    def clear(self) -> None:
        """Clear the cache."""
        self.cache = {}
        self.save_cache(overwrite=True)

    def save_cache(self, overwrite: bool = False) -> None:
        """Save the cache to disk."""
        logger.info(f"Saving cache to {self.cache_file}")
        file_exists = os.path.exists(self.cache_file)

        if overwrite or not file_exists:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f)
