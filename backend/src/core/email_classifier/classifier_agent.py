import ast
import json

import tiktoken
import logging
from backend.src.config.dev_config import *
from backend.src.core.email_classifier.calculate_confidence import CalculateConfidence
from backend.src.core.email_classifier.classifier import Classifier
from backend.src.core.email_classifier.agent_initializer import AgentInitialization
from backend.src.core.email_classifier.trigger_message import TriggerMessage
from backend.src.core.email_classifier.validator import Validator

logger = logging.getLogger(__name__)


class EmailClassifierProcessor:
    def __init__(self):
        self.model_name = AZURE_OPENAI_DEPLOYMENT_NAME
        self.max_input_tokens = MAX_INPUT_TOKEN
        self.agent_initializer = AgentInitialization()
        self.classifier = Classifier(self.agent_initializer)
        self.validator = Validator(self.agent_initializer)
        self.confidence_score = CalculateConfidence(self.agent_initializer)
        self.trigger_message = TriggerMessage(self.agent_initializer)


    def _truncate_to_max_tokens(self, text: str) -> tuple[str, bool]:
        try:
            enc = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        tokens = enc.encode(text)
        was_truncated = len(tokens) > self.max_input_tokens
        if was_truncated:
            tokens = tokens[:self.max_input_tokens]

        return enc.decode(tokens), was_truncated

    def get_response(self,result):
        classification_dict = ast.literal_eval(result)
        return classification_dict["response"]

    async def process_email(self, subject: str, body: str) -> str:
        combined_input = f"Subject: {subject}\n\nBody: {body}"
        truncated_input, was_truncated = self._truncate_to_max_tokens(combined_input)

        if was_truncated:
            logger.warning("Input was truncated due to token limit.")
            return await self.trigger_message.get_trigger_message(subject, body, "Unclear", 0.0, "Invalid")

        classification = await self.classifier.classify(truncated_input)
        classification_response = self.get_response(classification)
        validation = await self.validator.validate_classification(subject, body, classification_response)
        confidence = await self.confidence_score.calculate_confidence(subject, body, classification_response)

        if validation == "Invalid" and confidence < 95 and classification.lower() == "unclear":
            return await self.trigger_message.get_trigger_message(subject, body, classification_response, confidence, validation)

        return classification_response

    async def classify_via_vlm(self, base64_str: str) -> str:
        if not base64_str or not isinstance(base64_str, str):
            raise ValueError("Input base64 string is invalid or empty.")

        task = f"""
        Image (base64-encoded JPEG):
        data:image/jpeg;base64,{base64_str}
        """
        try:
            result = await self.agent_initializer.classification_with_llm.run(task=task)
            assistant_reply = next(
                (msg.content for msg in result.messages if msg.source == VLM_CLASSIFICATION_AGENT_NAME),
                None
            )
            if not assistant_reply:
                logger.warning("No content returned by the VLM model.")
                raise RuntimeError("No content extracted from image.")
            return assistant_reply
        except Exception as e:
            logger.exception("❌ OCR extraction via Azure OpenAI failed.")
            raise RuntimeError(f"OCR Extraction Failed: {e}") from e
