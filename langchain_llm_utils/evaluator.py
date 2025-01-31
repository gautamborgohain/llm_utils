from typing import Optional, List, Dict, Any, Protocol, Type, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from langchain_llm_utils.llm import LLM, ModelProvider
from langchain_llm_utils.common import BaseModelType, get_logger, decider
from langchain_core.messages import AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain.evaluation import embedding_distance, load_evaluator, EmbeddingDistance
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
import asyncio
import threading
from uuid import UUID

logger = get_logger("Evaluator")


class EvaluationType(Enum):
    HEURISTIC = "heuristic"
    LLM_JUDGE = "llm_judge"


class EvaluationScore(Protocol):

    @classmethod
    def get_string_response(cls, response: Union[str, BaseModel, AIMessage]) -> str:
        if isinstance(response, AIMessage):
            response = response.content
        elif isinstance(response, BaseModel):
            response = response.model_dump_json()
        return response

    def evaluate(self, prompt: str, response: str) -> float:
        """Return a score between 0 and 1"""
        ...

    async def aevaluate(self, prompt: str, response: str) -> float:
        """Return a score between 0 and 1"""
        ...


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    score_name: str
    details: Dict[str, Any] = None


class BaseHeuristicScore(EvaluationScore):
    def __init__(self, description: str):
        self.description = description

    @abstractmethod
    def evaluate(self, prompt: str, response: str, **kwargs) -> float:
        """
        Return a score between 0 and 1
        NOTE: Call response.get_string_response() to get the string response
        """
        pass

    @abstractmethod
    async def aevaluate(self, prompt: str, response: str, **kwargs) -> float:
        """
        Return a score between 0 and 1
        NOTE: Call response.get_string_response() to get the string response
        """
        pass


class BaseLLMJudgeScore(EvaluationScore):
    def __init__(
        self,
        llm_provider: ModelProvider,
        llm_name: str,
        response_model: Type[BaseModelType],
        description: str,
        llm_judge_type: str = "llm_judge",
    ):
        self.llm_provider = llm_provider
        self.llm_name = llm_name
        self.response_model = response_model
        self.description = description
        self.llm_judge = LLM(
            model_provider=self.llm_provider,
            model_name=self.llm_name,
            response_model=self.response_model,
            model_type=llm_judge_type,
            langfuse_tags=["llm_judge", llm_judge_type],
        )

    @abstractmethod
    def evaluate(self, prompt: str, response: str, **kwargs) -> float:
        """
        Return a score between 0 and 1
        NOTE: Call response.get_string_response() to get the string response
        """
        pass

    @abstractmethod
    async def aevaluate(self, prompt: str, response: str, **kwargs) -> float:
        """
        Return a score between 0 and 1
        NOTE: Call response.get_string_response() to get the string response
        """
        pass


class ResponseLengthScore(BaseHeuristicScore):
    """
    Length of the response.
    """

    def __init__(self):
        description = "Length of the response"
        super().__init__(description)

    def evaluate(
        self, prompt: str, response: Union[str, BaseModel, AIMessage], **kwargs
    ) -> float:
        response = self.get_string_response(response)
        return len(response)

    async def aevaluate(
        self, prompt: str, response: Union[str, BaseModel, AIMessage], **kwargs
    ) -> float:
        response = self.get_string_response(response)
        return len(response)


class CoherenceResponse(BaseModel):
    coherence: float = Field(
        description="The coherence of the response", default_factory=lambda: 0.0
    )


class CoherenceScore(BaseLLMJudgeScore):
    """
    Score the coherence of the response given the prompt.
    """

    def __init__(self, llm_provider: ModelProvider, llm_name: str):
        description = "Score the coherence of the response given the prompt"
        super().__init__(
            llm_provider,
            llm_name,
            response_model=CoherenceResponse,
            description=description,
            llm_judge_type="coherence_judge",
        )
        self.evaluation_judge_prompt = """
        Rate the coherence of this response to the prompt on a scale of 0-1.
        Return in JSON format:
        {{
            "coherence": "score (0-1)"
        }}
        
        Prompt: {prompt}
        Response: {response}
        
        Score: 
        """

    def evaluate(
        self, prompt: str, response: Union[str, BaseModel, AIMessage], **kwargs
    ) -> float:
        response = self.get_string_response(response)
        result = self.llm_judge.generate(
            template=self.evaluation_judge_prompt,
            input_variables={"prompt": prompt, "response": response},
        )

        return float(result.coherence)

    async def aevaluate(
        self, prompt: str, response: Union[str, BaseModel, AIMessage], **kwargs
    ) -> float:
        response = self.get_string_response(response)
        result = await self.llm_judge.agenerate(
            template=self.evaluation_judge_prompt,
            input_variables={"prompt": prompt, "response": response},
        )

        return float(result.coherence)


class EmbeddingDistanceScore(BaseHeuristicScore):
    """
    Simple evaluator that scores the embedding distance between the prompt and response.

    Provider:
    - HuggingFace
    - OpenAI

    Models examples:
    - HuggingFace: sentence-transformers/all-mpnet-base-v2
    - OpenAI: text-embedding-3-small
    """

    def __init__(
        self,
        embedding_model_provider: ModelProvider = ModelProvider.HUGGINGFACE,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        distance_metric: EmbeddingDistance = EmbeddingDistance.COSINE,
    ):
        description = "Score the embedding distance between the prompt and response"
        self.embedding_model_name = embedding_model_name
        if embedding_model_provider == ModelProvider.HUGGINGFACE:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name
            )
        elif embedding_model_provider == ModelProvider.OPENAI:
            self.embedding_model = OpenAIEmbeddings(model=self.embedding_model_name)
        else:
            raise ValueError(
                f"Unsupported embedding model provider: {embedding_model_provider}"
            )
        self.evaluator = load_evaluator(
            "embedding_distance",
            embeddings=self.embedding_model,
            distance_metric=distance_metric,
        )
        super().__init__(description)

    def _parse_score(self, response: dict) -> float:
        return float(response["score"])

    def evaluate(self, prompt: str, response: str, **kwargs) -> float:
        response = self.get_string_response(response)
        res = self.evaluator.evaluate_strings(prediction=response, reference=prompt)
        return self._parse_score(res)

    async def aevaluate(self, prompt: str, response: str, **kwargs) -> float:
        response = self.get_string_response(response)
        res = await self.evaluator.aevaluate_strings(
            prediction=response, reference=prompt
        )
        return self._parse_score(res)


class AlternateLLMResonseEmbeddingDistanceScore(BaseLLMJudgeScore):
    """
    Evaluator that scores the embedding distance between the response
    and a response generate by an alternate (ideally larger / better) LLM.

    """

    def __init__(
        self,
        llm_provider: ModelProvider,
        llm_name: str,
        expected_response_model: Type[BaseModelType],
        embedding_model_provider: ModelProvider = ModelProvider.HUGGINGFACE,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        distance_metric: EmbeddingDistance = EmbeddingDistance.COSINE,
    ):
        description = f"Score the embedding distance between the response and response generated by {llm_provider.value} {llm_name}"
        self.embedding_model_name = embedding_model_name
        if embedding_model_provider == ModelProvider.HUGGINGFACE:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name
            )
        elif embedding_model_provider == ModelProvider.OPENAI:
            self.embedding_model = OpenAIEmbeddings(model=self.embedding_model_name)
        else:
            raise ValueError(
                f"Unsupported embedding model provider: {embedding_model_provider}"
            )
        self.evaluator = load_evaluator(
            "embedding_distance",
            embeddings=self.embedding_model,
            distance_metric=distance_metric,
        )
        super().__init__(
            llm_provider,
            llm_name,
            response_model=expected_response_model,
            description=description,
            llm_judge_type="alternate_llm_responder",
        )

    def _parse_score(self, response: dict) -> float:
        return float(response["score"])

    def evaluate(self, prompt: str, response: str, **kwargs) -> float:
        response = self.get_string_response(response)
        judge_llm_response = self.llm_judge.generate(prompt=prompt)
        judge_llm_response = self.get_string_response(judge_llm_response)
        res = self.evaluator.evaluate_strings(
            prediction=response, reference=judge_llm_response
        )
        return self._parse_score(res)

    async def aevaluate(self, prompt: str, response: str, **kwargs) -> float:
        response = self.get_string_response(response)
        judge_llm_response = await self.llm_judge.agenerate(prompt=prompt)
        judge_llm_response = self.get_string_response(judge_llm_response)
        res = await self.evaluator.aevaluate_strings(
            prediction=response, reference=judge_llm_response
        )
        return self._parse_score(res)


class EvaluatorConfig:
    """
    Setup configuration for the Evaluator.
    Use this define the list of evaluations you want to perform on LLM responses.

    Example Usage:
    -------------
    ```python
    evaluator_config = EvaluatorConfig()
    evaluator_config.add_heuristic(ResponseLengthScore())
    evaluator_config.add_llm_judge(CoherenceScore(llm_provider=ModelProvider.OPENAI, llm_name="gpt-4o"))
    ```
    """

    def __init__(self):
        self.scores: List[EvaluationScore] = []

    def add_heuristic(self, score: BaseHeuristicScore) -> "EvaluatorConfig":
        self.scores.append(score)
        return self

    def add_llm_judge(self, score: BaseLLMJudgeScore) -> "EvaluatorConfig":
        self.scores.append(score)
        return self


class Evaluator:
    """
    Evaluate LLM responses with optional sampling based on run_id.

    Example Usage:
    -------------
    ```python
    evaluator = Evaluator(sample_rate=0.1)
    evaluator.evaluate("What is the capital of France?", "Paris")
    ```

    With Langchain:
    --------------
    ```python
    evaluator = Evaluator(sample_rate=0.1)
    chat = ChatOpenAI(
        model_name="gpt-4",
        callbacks=evaluator.callbacks,
        openai_api_key="fake-key",
    )
    ```
    """

    def __init__(
        self, config: Optional[EvaluatorConfig] = None, sample_rate: float = 1.0
    ):
        self.config = config or EvaluatorConfig()
        self.sample_rate = sample_rate
        self.evaluation_results = []  # Store evaluation results
        self.callback = EvaluatorCallback(self)
        self.callbacks = [self.callback]  # Store for external access

    def _should_evaluate(self, run_id: Optional[UUID] = None) -> bool:
        """Determine if evaluation should be performed based on sampling."""
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        if run_id is None:
            return True  # Always evaluate if no run_id provided
        should_evaluate = decider(run_id, self.sample_rate, "evaluator")
        logger.debug(
            f"Input run_id: {run_id}, should evaluate: {should_evaluate}, sample_rate: {self.sample_rate}"
        )
        return should_evaluate

    def evaluate(
        self, prompt: str, response: str, run_id: Optional[UUID] = None
    ) -> List[EvaluationResult]:
        """Evaluate with optional sampling based on run_id."""

        # Check if we should evaluate based on sampling
        if not self._should_evaluate(run_id):
            return []

        results = []
        for score in self.config.scores:
            try:
                score_value = score.evaluate(prompt, response)
                result = EvaluationResult(
                    score=score_value,
                    score_name=score.__class__.__name__,
                    details={"description": score.description},
                )
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Error evaluating score {score.__class__.__name__}: {e}",
                    exc_info=True,
                )

        # Store the results
        self.evaluation_results.append(results)
        return results

    async def aevaluate(
        self, prompt: str, response: str, run_id: Optional[UUID] = None
    ) -> List[EvaluationResult]:
        """Async evaluate with optional sampling based on run_id."""

        # Check if we should evaluate based on sampling
        if not self._should_evaluate(run_id):
            return []

        results = []
        for score in self.config.scores:
            try:
                result = await score.aevaluate(prompt, response)
                results.append(
                    EvaluationResult(
                        score=result,
                        score_name=score.__class__.__name__,
                        details={"description": score.description},
                    )
                )
            except Exception as e:
                logger.error(
                    f"Error evaluating score {score.__class__.__name__}: {e}",
                    exc_info=True,
                )
        return results


class EvaluatorCallback(BaseCallbackHandler):
    """Callback handler that tracks run IDs for evaluation sampling."""

    def __init__(self, evaluator: Evaluator):
        self._thread_local = threading.local()
        self.evaluator = evaluator

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Store the current run_id and prompt."""
        if not hasattr(self._thread_local, "current_run_id"):
            self._thread_local.current_run_id = None
        self._thread_local.current_run_id = run_id
        self._thread_local.current_prompt = prompts[0] if prompts else None

    def on_llm_end(
        self,
        response: Any,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Trigger evaluation when LLM response is received."""
        prompt = getattr(self._thread_local, "current_prompt", None)
        if prompt and hasattr(response, "generations"):
            # Trigger evaluation
            self.evaluator.evaluate(prompt, response, run_id=run_id)


class BasicEvaluationSuite(Evaluator):
    def __init__(self, sample_rate: float = 1.0):
        config = EvaluatorConfig()
        config.add_heuristic(ResponseLengthScore())
        config.add_llm_judge(
            CoherenceScore(
                llm_provider=ModelProvider.OPENAI,
                llm_name="gpt-4o",
            )
        )
        super().__init__(config=config, sample_rate=sample_rate)
