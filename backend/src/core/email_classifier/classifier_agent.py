import json
import tiktoken
import logging

from backend.src.config.dev_config import (
    AZURE_OPENAI_DEPLOYMENT_NAME, MAX_INPUT_TOKEN, EMAIL_CLASSIFIER_AGENT_NAME,
    VALIDATOR_AGENT_NAME, CONFIDENCE_AGENT_NAME, TIGGER_REASON_AGENT_NAME, VLM_CLASSIFICATION_AGENT_NAME
)
from backend.src.core.base_agents.base_agent import BaseAgent
from backend.src.core.base_client.base_client import OpenAiClient
from backend.src.prompts.confidence_prompt import CONFIDENCE_PROMPT
from backend.src.prompts.emailclassifer_prompt import CLASSIFICATION_PROMPT, CLASSIFICATION_PROMPT_VLM
from backend.src.prompts.trigger_reason_prompt import TRIGGER_REASON_PROMPT
from backend.src.prompts.validator_prompt import VALIDATION_PROMPT

logger = logging.getLogger(__name__)


class EmailClassifierProcessor:
    def __init__(self):
        self.client = OpenAiClient()
        self.base_agent = BaseAgent(self.client.open_ai_chat_completion_client)
        self.model_name = AZURE_OPENAI_DEPLOYMENT_NAME
        self.max_input_tokens = MAX_INPUT_TOKEN

        self._initialize_agents()

    def _initialize_agents(self):
        self.email_classifier_agent = self.base_agent.create_assistant_agent(
            name=EMAIL_CLASSIFIER_AGENT_NAME,
            prompt=CLASSIFICATION_PROMPT
        )
        self.validator_agent = self.base_agent.create_assistant_agent(
            name=VALIDATOR_AGENT_NAME,
            prompt=VALIDATION_PROMPT
        )
        self.confidence_agent = self.base_agent.create_assistant_agent(
            name=CONFIDENCE_AGENT_NAME,
            prompt=CONFIDENCE_PROMPT
        )
        self.trigger_reason_agent = self.base_agent.create_assistant_agent(
            name=TIGGER_REASON_AGENT_NAME,
            prompt=TRIGGER_REASON_PROMPT
        )
        self.classification_with_llm = self.base_agent.create_assistant_agent(
            name=VLM_CLASSIFICATION_AGENT_NAME,
            prompt=CLASSIFICATION_PROMPT_VLM
        )

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

    async def _run_agent_task(self, agent, task: str, fallback_result=None, result_type=str):
        try:
            result = await agent.run(task=task)
            content = list(result.messages)[-1].content
            return result_type(content)
        except Exception as e:
            logger.exception(f"❌ Agent task failed: {agent.name}")
            return fallback_result

    async def classify_email(self, message: str) -> str:
        return await self._run_agent_task(self.email_classifier_agent, message, fallback_result="Unclear")

    async def validate_classification(self, subject: str, body: str, classification: str) -> str:
        task = f"Email Subject: {subject}\nEmail Body: {body}\nClassification Given: {classification}"
        return await self._run_agent_task(self.validator_agent, task, fallback_result="Invalid")

    async def calculate_confidence(self, subject: str, body: str, classification: str) -> float:
        task = f"Email Subject: {subject}\nEmail Body: {body}\nClassification Given: {classification}"
        try:
            result = await self.confidence_agent.run(task=task)
            content = list(result.messages)[-1].content
            parsed = json.loads(content)
            return float(parsed.get("confidence", 0.0))
        except Exception:
            logger.exception("❌ Failed to calculate confidence.")
            return 0.0

    async def get_trigger_reason(self, subject: str, body: str, classification: str, confidence: float, validation: str) -> str:
        task = f"""Subject: {subject}
            Body: {body}
            Classification: {classification}
            Confidence: {confidence}
            Validation Result: {validation}"""
        try:
            result = await self.trigger_reason_agent.run(task=task)
            content = list(result.messages)[-1].content
            return json.loads(content).get("reason", "Unknown reason.")
        except Exception:
            logger.exception("❌ Failed to extract trigger reason.")
            return "Reason could not be determined due to an internal error."

    async def trigger_message(self, subject: str, body: str, classification: str, confidence: float, validation: str = "Invalid") -> str:
        reason = await self.get_trigger_reason(subject, body, classification, confidence, validation)
        return f"Message Triggered, Reason: {reason}"

    async def process_email(self, subject: str, body: str) -> str:
        combined_input = f"Subject: {subject}\n\nBody: {body}"
        truncated_input, was_truncated = self._truncate_to_max_tokens(combined_input)

        if was_truncated:
            logger.warning("Input was truncated due to token limit.")
            return await self.trigger_message(subject, body, "Unclear", 0.0, "Invalid")

        classification = await self.classify_email(truncated_input)
        validation = await self.validate_classification(subject, body, classification)
        confidence = await self.calculate_confidence(subject, body, classification)

        if validation == "Invalid" and confidence < 95 and classification.lower() == "unclear":
            return await self.trigger_message(subject, body, classification, confidence, validation)

        return classification

    async def classify_via_vlm(self, base64_str: str) -> str:
        if not base64_str or not isinstance(base64_str, str):
            raise ValueError("Input base64 string is invalid or empty.")

        task = f"""
        Image (base64-encoded JPEG):
        data:image/jpeg;base64,{base64_str}
        """
        try:
            result = await self.classification_with_llm.run(task=task)
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
