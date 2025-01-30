from typing import Type, Optional, Any, Union, Dict, Generic, List
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from langchain_llm_utils.config import config
from langchain_llm_utils.common import get_logger, BaseModelType
from langchain_llm_utils.request_cache import RequestCache
from langchain_llm_utils.rate_limiter import default_rate_limiter
from httpx import ConnectError
from abc import ABC, abstractmethod
from enum import Enum
from langchain_core.messages import AIMessage, BaseMessage
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import os
import asyncio

logger = get_logger("LLM")
langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "false") == "true"

if not langfuse_enabled:
    os.environ["LANGFUSE_SECRET_KEY"] = ""
    os.environ["LANGFUSE_PUBLIC_KEY"] = ""
    os.environ["LANGFUSE_HOST"] = ""

langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    enabled=langfuse_enabled,
)


class ModelProvider(Enum):
    VERTEX_AI = "vertex_ai"
    OLLAMA = "ollama"
    OPENAI = "openai"


class ModelConfig(BaseModel):
    provider: ModelProvider
    base_model: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: Optional[int] = None
    api_base: Optional[str] = None
    rate_limiter: Optional[Any] = default_rate_limiter
    structured_output_enabled: bool = True
    application_type: Optional[str] = "LLM"

    def validate_temperature(self):
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return True


class LLMProvider(ABC):
    @abstractmethod
    def invoke(self, prompt: str) -> BaseMessage:
        pass

    @abstractmethod
    async def ainvoke(self, prompt: str) -> BaseMessage:
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

    async def ainvoke(self, prompt: str) -> str:
        return await self.client.ainvoke(prompt)

    def with_structured_output(
        self, response_model: Type[BaseModelType]
    ) -> "OllamaProvider":
        self.response_model = response_model
        if self.model_config.structured_output_enabled:
            logger.info(
                f"Setting structured output mode for {self.model_config.base_model} for {self.model_config.application_type}"
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

    async def ainvoke(self, prompt: str) -> str:
        return await self.client.ainvoke(prompt)

    def with_structured_output(
        self, response_model: Type[BaseModelType]
    ) -> "VertexAIProvider":
        self.response_model = response_model
        logger.info(
            f"Setting structured output mode for {self.model_config.base_model} for {self.model_config.application_type}"
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

    def invoke(self, prompt: str) -> BaseMessage:
        return self.client.invoke(prompt)

    async def ainvoke(self, prompt: str) -> BaseMessage:
        return await self.client.ainvoke(prompt)

    def with_structured_output(
        self, response_model: Type[BaseModelType]
    ) -> "OpenAIProvider":
        self.response_model = response_model
        logger.info(
            f"Setting structured output mode for {self.model_config.base_model} for {self.model_config.application_type}"
        )
        self.client = self.client.with_structured_output(response_model)
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
        application_type: Optional[str] = None,
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
                application_type=application_type,
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
                application_type=application_type,
            )
        elif model_provider == ModelProvider.OPENAI:
            return ModelConfig(
                provider=ModelProvider.OPENAI,
                base_model=model_name,
                temperature=temperature,
                rate_limiter=rate_limiter,
                api_base=config.OPENAI_API_BASE,
                application_type=application_type,
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
        evaluator: Optional["Evaluator"] = None,  # type: ignore
        langfuse_tags: Optional[List[str]] = None,
    ):
        self.model_provider = model_provider
        self.model_name = model_name
        self.config = LLMFactory.create_config(
            model_provider=self.model_provider,
            model_name=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            rate_limiter=rate_limiter,
            application_type=model_type,
        )
        self.cache = cache
        self.response_model = response_model
        self.model_type = model_type
        self.evaluator = evaluator
        self.langfuse_tags = langfuse_tags
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

    @observe(as_type="generation")
    def invoke(self, prompt: str) -> Union[str, BaseModelType]:
        """Invoke the LLM with a prompt."""
        try:
            res = self._llm.invoke(prompt)

            # Store the current observation ID
            observation_id = langfuse_context.get_current_observation_id()
            trace_id = langfuse_context.get_current_trace_id()
            logger.debug(
                f"Current observation ID: {observation_id}, trace ID: {trace_id}, model name: {self.model_name}, model type: {self.model_type}"
            )
            langfuse_context.update_current_observation(
                input=prompt,
                model=self.model_name,
                output=res,
                name=self.model_type,
            )

            # Run evaluations if evaluator exists
            if self.evaluator:
                self._run_evaluations(
                    prompt, res, trace_id, observation_id, run_id=trace_id
                )

            return res
        except (ConnectionRefusedError, ConnectionError, ConnectError) as e:
            logger.error(f"Connection refused invoking LLM {self.model_name}: {e}")
            raise type(e)(
                f"Failed to connect to LLM {self.model_name}. Please check if the service is running and accessible."
            )
        except Exception as e:
            logger.error(f"Error invoking LLM {self.model_name}: {e}")
            return None

    @observe(as_type="generation")
    async def ainvoke(self, prompt: str) -> Union[str, BaseModelType]:
        """Async version of invoke method."""
        try:
            res = await self._llm.ainvoke(prompt)

            # Store the current observation ID
            observation_id = langfuse_context.get_current_observation_id()
            trace_id = langfuse_context.get_current_trace_id()
            logger.debug(
                f"AInvoke Current observation ID: {observation_id}, trace ID: {trace_id}, model name: {self.model_name}, model type: {self.model_type}"
            )
            langfuse_context.update_current_observation(
                input=prompt,
                model=self.model_name,
                output=res,
                name=self.model_type,
            )

            # Run evaluations if evaluator exists
            if self.evaluator:
                asyncio.create_task(
                    self._arun_evaluations(
                        prompt, res, trace_id, observation_id, run_id=trace_id
                    )
                )

            return res
        except (ConnectionRefusedError, ConnectionError, ConnectError) as e:
            logger.error(f"Connection refused invoking LLM {self.model_name}: {e}")
            raise type(e)(
                f"Failed to connect to LLM {self.model_name}. Please check if the service is running and accessible."
            )
        except Exception as e:
            logger.error(f"Error invoking LLM {self.model_name}: {e}")
            return None

    def _run_evaluations(
        self,
        prompt: str,
        response: Union[str, BaseModelType],
        trace_id: str,
        observation_id: str,
        run_id: str,
    ):
        """Run evaluations synchronously and update Langfuse scores."""
        try:
            logger.debug(
                f"Running evaluations for trace {trace_id} and observation {observation_id}"
            )
            evaluations = self.evaluator.evaluate(prompt, response, run_id)
            logger.debug(
                f"Pushing evaluations to Langfuse for trace {trace_id} and observation {observation_id}"
            )
            for eval in evaluations:
                langfuse.score(
                    trace_id=trace_id,
                    observation_id=observation_id,
                    value=eval.score,
                    name=eval.score_name,
                    comment=eval.details["description"] if eval.details else "",
                )
        except Exception as e:
            logger.error(f"Error running evaluations in LLM: {e}")

    async def _arun_evaluations(
        self,
        prompt: str,
        response: Union[str, BaseModelType],
        trace_id: str,
        observation_id: str,
        run_id: str,
    ):
        """Run evaluations asynchronously and update Langfuse scores."""
        try:
            logger.debug(
                f"Running evaluations for trace {trace_id} and observation {observation_id}"
            )
            evaluations = await self.evaluator.aevaluate(prompt, response, run_id)
            logger.debug(
                f"Pushing evaluations to Langfuse for trace {trace_id} and observation {observation_id}"
            )
            for eval in evaluations:
                langfuse.score(
                    trace_id=trace_id,
                    observation_id=observation_id,
                    value=eval.score,
                    name=eval.score_name,
                    comment=eval.details["description"] if eval.details else "",
                )
        except Exception as e:
            logger.error(f"Error running evaluations in LLM: {e}")

    def _handle_cached_response(
        self,
        final_prompt: str,
    ) -> Optional[BaseModelType]:
        """Handle cached response retrieval and storage."""
        cache_key = self.cache.get_cache_key(
            model_name=self.model_name,
            model_provider=self.model_provider,
            final_prompt=final_prompt,
        )
        # Try to get from cache
        return self.cache.get_parsed(cache_key, self.response_model)

    def _cache_response(self, final_prompt: str, response: Any) -> None:
        """Cache the response if it exists."""
        if response:
            cache_key = self.cache.get_cache_key(
                model_name=self.model_name,
                model_provider=self.model_provider,
                final_prompt=final_prompt,
            )
            self.cache.set(cache_key, response)

    def _process_structured_response(self, res: Any) -> Union[str, BaseModelType]:
        """Process and validate structured response."""
        if not self.response_model:
            return res

        # Return empty response model if no result
        if not res:
            return self.response_model()

        # Handle structured output validation
        if not isinstance(res, self.response_model):
            try:
                if isinstance(res, AIMessage):
                    res = res.content
                res = self.response_model.model_validate_json(res)
            except ValidationError as e:
                logger.error(f"Failed to validate structured output: {e}")
                return self.response_model()

        return res

    @observe()
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

        langfuse_context.update_current_observation(
            model=self.model_name,
            name=self.model_type,
            tags=self.langfuse_tags,
        )

        # Handle cached responses if caching is enabled
        if self.response_model and self.cache:
            cached_response = self._handle_cached_response(final_prompt)
            if cached_response is not None:
                return cached_response

            response = self.invoke(final_prompt)
            self._cache_response(final_prompt, response)
            return response

        # Generate and process response
        res = self.invoke(final_prompt)
        return self._process_structured_response(res)

    @observe()
    async def agenerate(
        self,
        prompt: str = "",
        template: Optional[str] = None,
        input_variables: Optional[Dict[str, Any]] = None,
    ) -> Union[str, BaseModelType]:
        """
        Async version of generate method.
        """
        final_prompt = self._format_prompt(prompt, template, input_variables)
        langfuse_context.update_current_observation(
            model=self.model_name,
            name=self.model_type,
            tags=self.langfuse_tags,
        )

        # Handle cached responses if caching is enabled
        if self.response_model and self.cache:
            cached_response = self._handle_cached_response(final_prompt)
            if cached_response is not None:
                return cached_response

            response = await self.ainvoke(final_prompt)
            self._cache_response(final_prompt, response)
            return response

        # Generate and process response
        res = await self.ainvoke(final_prompt)
        return self._process_structured_response(res)
