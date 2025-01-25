from typing import Type, Optional, Any, Union, Dict, Generic
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from llm_utils.config import config
from llm_utils.common import logger, BaseModelType
from llm_utils.request_cache import RequestCache
from llm_utils.rate_limiter import default_rate_limiter
from httpx import ConnectError
from abc import ABC, abstractmethod
from enum import Enum
from langchain_core.messages import AIMessage


class ModelProvider(Enum):
    VERTEX_AI = "vertex_ai"
    OLLAMA = "ollama"
    OPENAI = "openai"


class ModelConfig(BaseModel):
    provider: ModelProvider
    base_model: str
    temperature: float = Field(ge=0.0, le=1.0)
    max_tokens: Optional[int] = None
    api_base: Optional[str] = None
    rate_limiter: Optional[Any] = default_rate_limiter
    structured_output_enabled: bool = True

    def validate_temperature(self):
        if not 0 <= self.temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return True


class LLMProvider(ABC):
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        pass

    @abstractmethod
    def with_structured_output(
        self, response_model: Type[BaseModelType]
    ) -> "LLMProvider":
        pass


class OllamaProvider(LLMProvider):
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.client = ChatOllama(
            model=model_config.base_model,
            temperature=model_config.temperature,
            base_url=model_config.api_base,
            max_tokens=model_config.max_tokens,
            rate_limiter=model_config.rate_limiter,
        )

    def invoke(self, prompt: str) -> str:
        return self.client.invoke(prompt)

    def with_structured_output(
        self, response_model: Type[BaseModelType]
    ) -> "OllamaProvider":
        self.response_model = response_model
        if self.model_config.structured_output_enabled:
            logger.info(
                f"Setting structured output mode for {self.model_config.base_model}"
            )
            self.client = self.client.with_structured_output(response_model)
        return self


class VertexAIProvider(LLMProvider):
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.client = ChatVertexAI(
            model_name=model_config.base_model,
            project=config.GOOGLE_CLOUD_PROJECT,
            temperature=model_config.temperature,
            rate_limiter=model_config.rate_limiter,
        )

    def invoke(self, prompt: str) -> str:
        return self.client.invoke(prompt)

    def with_structured_output(
        self, response_model: Type[BaseModelType]
    ) -> "VertexAIProvider":
        self.response_model = response_model
        logger.info(
            f"Setting structured output mode for {self.model_config.base_model}"
        )
        self.client = self.client.with_structured_output(response_model)
        return self


class OpenAIProvider(LLMProvider):
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.client = ChatOpenAI(
            model=model_config.base_model,
            temperature=model_config.temperature,
            rate_limiter=model_config.rate_limiter,
            base_url=model_config.api_base,
        )

    def invoke(self, prompt: str) -> str:
        return self.client.invoke(prompt)

    def with_structured_output(
        self, response_model: Type[BaseModelType]
    ) -> "OpenAIProvider":
        self.response_model = response_model
        return self


class LLMFactory:
    """Initialize the appropriate LLM based on the model name.

    Currently supports:
    - Vertex AI (provided GOOGLE_CLOUD_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS are available in the environment)
    - Ollama (provided OLLAMA_URL is available in the environment)
    - OpenAI (provided OPENAI_API_KEY is available in the environment)
    """

    _registry = {
        ModelProvider.OLLAMA: OllamaProvider,
        ModelProvider.OPENAI: OpenAIProvider,
        ModelProvider.VERTEX_AI: VertexAIProvider,
    }

    @classmethod
    def create_config(
        cls,
        model_provider: ModelProvider,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        rate_limiter: Any = default_rate_limiter,
    ) -> ModelConfig:
        """Create model config from model string.

        Example model strings:
        - gemini-1.5-flash
        - ollama:llama3.1
        - ollama:llama3.1:70b
        - openai:gpt-4o

        Custom gguf models on Ollama are not supported for structured output.
        """
        custom_gguf_models = config.CUSTOM_GGUF_MODELS
        if model_provider == ModelProvider.VERTEX_AI:
            return ModelConfig(
                provider=ModelProvider.VERTEX_AI,
                base_model=model_name,
                temperature=temperature,
                rate_limiter=rate_limiter,
            )
        elif model_provider == ModelProvider.OLLAMA:
            base_model = model_name
            return ModelConfig(
                provider=ModelProvider.OLLAMA,
                base_model=base_model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_base=config.OLLAMA_URL,
                rate_limiter=rate_limiter,
                structured_output_enabled=(
                    False if base_model in custom_gguf_models else True
                ),
            )
        elif model_provider == ModelProvider.OPENAI:
            return ModelConfig(
                provider=ModelProvider.OPENAI,
                base_model=model_name,
                temperature=temperature,
                rate_limiter=rate_limiter,
                api_base=config.OPENAI_API_BASE,
            )
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")

    @classmethod
    def create_provider(cls, model_config: ModelConfig) -> LLMProvider:
        """Create LLM provider instance based on config."""
        provider_class = cls._registry.get(model_config.provider)
        if not provider_class:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
        return provider_class(model_config)


class LLM(Generic[BaseModelType]):
    """
    A unified interface for interacting with different LLM implementations.
    Supports both structured and unstructured outputs, caching, and temperature control.

    This class provides a simple way to:
    1. Generate responses from different LLM providers (Vertex AI, Ollama)
    2. Handle both structured (Pydantic models) and unstructured (string) outputs
    3. Cache responses for faster repeated queries
    4. Control temperature and other model parameters

    Example usage:

    ```python
    # Simple string output
    llm = LLM(model_provider='vertex-ai', model_name="gemini-1.5-flash", temperature=0.2)
    response = llm.generate("Tell me a joke")

    # With template
    response = llm.generate(
        template="Tell me a {type} joke about {topic}",
        input_variables={"type": "dad", "topic": "programming"}
    )

    # Structured output with caching
    from pydantic import BaseModel

    class JokeResponse(BaseModel):
        setup: str
        punchline: str

    cache = RequestCache("jokes_cache.json")
    llm = LLM[JokeResponse](
        model="gemini-1.5-flash",
        cache=cache,
        response_model=JokeResponse
    )

    joke = llm.generate("Tell me a joke")
    print(f"{joke.setup} - {joke.punchline}")
    ```
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        model_name: str = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        cache: Optional[RequestCache] = None,
        response_model: Optional[Type[BaseModelType]] = None,
        model_type: Optional[str] = "invoke",
        rate_limiter: Optional[Any] = None,
    ):
        self.model_provider = model_provider
        self.model_name = model_name
        self.config = LLMFactory.create_config(
            model_provider=self.model_provider,
            model_name=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            rate_limiter=rate_limiter,
        )
        self.cache = cache
        self.response_model = response_model
        self.model_type = model_type
        self._llm = self._initialize_llm()

        if response_model and self._is_valid_response_model(response_model):
            self._llm = self._llm.with_structured_output(response_model)

    def _is_valid_response_model(self, response_model: Type[BaseModelType]) -> bool:
        """Check if the response model is valid by validating that all fields have default values."""
        try:
            return isinstance(response_model(), response_model)
        except Exception as e:
            logger.error(
                f"Error: {e} Make sure all fields in {response_model} have default values."
            )
            return False

    def _initialize_llm(self) -> LLMProvider:
        """Initialize the appropriate LLM provider based on config."""
        try:
            provider = LLMFactory.create_provider(self.config)
            return provider
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            raise

    def _format_prompt(
        self,
        prompt: str,
        template: Optional[str] = None,
        input_variables: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format the prompt using template if provided."""
        if template and input_variables:
            prompt_template = PromptTemplate(
                template=template, input_variables=list(input_variables.keys())
            )
            return prompt_template.format(**input_variables)
        return prompt

    def invoke(self, prompt: str) -> Union[str, BaseModelType]:
        """Invoke the LLM with a prompt."""
        try:
            res = self._llm.invoke(prompt)
            return res
        except (ConnectionRefusedError, ConnectionError, ConnectError) as e:
            logger.error(f"Connection refused invoking LLM {self.model_name}: {e}")
            raise type(e)(
                f"Failed to connect to LLM {self.model_name}. Please check if the service is running and accessible."
            )
        except Exception as e:
            logger.error(f"Error invoking LLM {self.model_name}: {e}")
            return None

    def generate(
        self,
        prompt: str = "",
        template: Optional[str] = None,
        input_variables: Optional[Dict[str, Any]] = None,
    ) -> Union[str, BaseModelType]:
        """
        Generate a response for a prompt or generate a response from a template.

        Args:
            prompt: Direct prompt string if template is None
            template: Optional template string with placeholders
            input_variables: Dict of variables to fill template placeholders

        Returns:
            Either a string response or an instance of response_model if specified
        """
        final_prompt = self._format_prompt(prompt, template, input_variables)

        if self.response_model and self.cache:
            # Try to get from cache
            cached_response = self.cache.get_parsed(final_prompt, self.response_model)
            if cached_response is not None:
                return cached_response

            # Generate new response
            response = self.invoke(final_prompt)

            # Cache the response
            if response:
                self.cache.set(final_prompt, response)
            else:
                if self.response_model:
                    return self.response_model()
            return response

        res = self.invoke(final_prompt)

        if self.response_model:
            # Check res matches response_model and if not, try to parse it as json and return as a response_model object
            if not isinstance(res, self.response_model):
                try:
                    if isinstance(res, AIMessage):
                        res = res.content
                    res = self.response_model.model_validate_json(res)
                except ValidationError as e:
                    logger.error(f"Failed to validate structured output: {e}")
                    return self.response_model()

        # Incase there is no response from LLM, return blank response model
        if not res and self.response_model:
            return self.response_model()
        return res
