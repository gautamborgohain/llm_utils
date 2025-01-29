from typing import Optional, List, Dict, Any, Protocol, Type, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from langchain_llm_utils.llm import LLM, ModelProvider
from langchain_llm_utils.common import BaseModelType, logger
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
import asyncio


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
    def __init__(self, config: Optional[EvaluatorConfig] = None):
        self.config = config or EvaluatorConfig()

    def evaluate(self, prompt: str, response: str) -> List[EvaluationResult]:
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
        return results

    async def aevaluate(self, prompt: str, response: str) -> List[EvaluationResult]:
        results = []
        tasks = []

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


class BasicEvaluationSuite:

    def __init__(self):
        self.config = EvaluatorConfig()
        self.config.add_heuristic(ResponseLengthScore())
        self.config.add_llm_judge(
            CoherenceScore(
                llm_provider=ModelProvider.OPENAI,
                llm_name="gpt-4o-mini",
            )
        )
        self.evaluator = Evaluator(config=self.config)

    def evaluate(self, prompt: str, response: str) -> List[EvaluationResult]:
        return self.evaluator.evaluate(prompt, response)

    async def aevaluate(self, prompt: str, response: str) -> List[EvaluationResult]:
        return await self.evaluator.aevaluate(prompt, response)
