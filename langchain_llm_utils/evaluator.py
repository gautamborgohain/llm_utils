from typing import Optional, List, Dict, Any, Protocol, Type, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from langchain_llm_utils.llm import LLM, ModelProvider
from langchain_llm_utils.common import BaseModelType, get_logger, decider
from langchain_core.messages import AIMessage
from langchain_core.callbacks import BaseCallbackHandler
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
    def evaluate(self, prompt: str, response: str) -> float:
        pass

    @abstractmethod
    async def aevaluate(self, prompt: str, response: str) -> float:
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
    def evaluate(self, prompt: str, response: str) -> float:
        pass

    @abstractmethod
    async def aevaluate(self, prompt: str, response: str) -> float:
        pass


class ResponseLengthScore(BaseHeuristicScore):
    """
    Length of the response.
    """

    def __init__(self):
        description = "Length of the response"
        super().__init__(description)

    def evaluate(
        self, prompt: str, response: Union[str, BaseModel, AIMessage]
    ) -> float:
        response = self.get_string_response(response)
        return len(response)

    async def aevaluate(
        self, prompt: str, response: Union[str, BaseModel, AIMessage]
    ) -> float:
        response = self.get_string_response(response)
        return len(response)


class CoherenceResponse(BaseModel):
    coherence: float = Field(
        description="The coherence of the response", default_factory=lambda: 0.0
    )


class CoherenceScore(BaseLLMJudgeScore):

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
        self, prompt: str, response: Union[str, BaseModel, AIMessage]
    ) -> float:
        response = self.get_string_response(response)
        result = self.llm_judge.generate(
            template=self.evaluation_judge_prompt,
            input_variables={"prompt": prompt, "response": response},
        )

        return float(result.coherence)

    async def aevaluate(
        self, prompt: str, response: Union[str, BaseModel, AIMessage]
    ) -> float:
        response = self.get_string_response(response)
        result = await self.llm_judge.agenerate(
            template=self.evaluation_judge_prompt,
            input_variables={"prompt": prompt, "response": response},
        )

        return float(result.coherence)


class EvaluatorConfig:
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
