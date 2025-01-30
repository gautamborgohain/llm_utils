from pydantic import BaseModel, Field
from langchain_llm_utils.llm import LLM, ModelProvider
from langchain_llm_utils.evaluator import BasicEvaluationSuite
from langchain_llm_utils.common import get_logger

logger = get_logger("TextAnalytics")

import argparse
import time
import asyncio


class LanguageDetected(BaseModel):
    language: str = Field(
        description="The language of the text in lowercase", default_factory=lambda: ""
    )


class Translation(BaseModel):
    translated_text: str = Field(
        description="The text after being translated to English",
        default_factory=lambda: "",
    )


class TranslationService:
    def __init__(self):
        self.evaluator = BasicEvaluationSuite(sample_rate=0.9)
        self.lang_detector = LLM(
            model_provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
            temperature=1.0,
            response_model=LanguageDetected,
            evaluator=self.evaluator,
            model_type="lang_detector",
            langfuse_tags=["text_analytics", "translation_service", "lang_detector"],
        )
        self.translator = LLM(
            model_provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
            temperature=1.0,
            response_model=Translation,
            evaluator=self.evaluator,
            model_type="translator",
            langfuse_tags=["text_analytics", "translation_service", "translator"],
        )
        self.lang_detector_prompt = """
        You are a language detector. You are given a text and you need to determine the language of the text.
        Language code can be like - en, es, fr, de, it, etc.
        Return in JSON format:
        {{
            "language": "language_code"
        }}
        
        The text is: {text}
        """
        self.translator_prompt = """
        You are a language translator. You are given a text and you need to translate it to English.
        
        Return in JSON format:
        {{
            "translated_text": "translated_text"
        }}
        
        The text is: {text}
        """


def translate_to_english(text: str) -> Translation:
    """Synchronous translation"""
    translation_service = TranslationService()

    lang_detected = translation_service.lang_detector.generate(
        template=translation_service.lang_detector_prompt,
        input_variables={"text": text},
    )
    logger.debug(f"Language detected: {lang_detected.language}")
    if lang_detected.language == "en":
        return Translation(translated_text=text)
    else:
        return translation_service.translator.generate(
            template=translation_service.translator_prompt,
            input_variables={"text": text},
        )


async def translate_to_english_async(text: str) -> Translation:
    """Asynchronous translation"""
    translation_service = TranslationService()

    # First detect language asynchronously
    lang_detected = await translation_service.lang_detector.agenerate(
        template=translation_service.lang_detector_prompt,
        input_variables={"text": text},
    )

    logger.debug(f"Language detected: {lang_detected.language}")
    if lang_detected.language == "en":
        return Translation(translated_text=text)
    else:
        # Then translate asynchronously
        return await translation_service.translator.agenerate(
            template=translation_service.translator_prompt,
            input_variables={"text": text},
        )


async def main_async():
    """
    Example command:
    poetry run python text_analytics.py "Hello, how are you?" --async
    """
    parser = argparse.ArgumentParser(description="Translate text to English")
    parser.add_argument("text", type=str, help="Text to translate")
    parser.add_argument("--async", action="store_true", help="Run asynchronously")
    args = parser.parse_args()

    if getattr(args, "async"):
        print("\n=== Running Asynchronous Translation ===")
        start_time = time.time()
        result = await translate_to_english_async(args.text)
        end_time = time.time()
        # Wait for evaluations to complete
        await asyncio.sleep(2)
        print(f"Async translated text: {result.translated_text}")
        print(f"Async time taken: {end_time - start_time:.2f} seconds")
    else:
        print("\n=== Running Synchronous Translation ===")
        start_time = time.time()
        result = translate_to_english(args.text)
        end_time = time.time()
        print(f"Sync translated text: {result.translated_text}")
        print(f"Sync time taken: {end_time - start_time:.2f} seconds")
        print("Waiting for sync evaluations to complete...")


if __name__ == "__main__":
    asyncio.run(main_async())
