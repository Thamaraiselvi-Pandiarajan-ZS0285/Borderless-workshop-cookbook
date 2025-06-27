import json
import tiktoken
import logging
from openai import AzureOpenAI
from backend.src.core.base_agents.base_agent import BaseAgent
from backend.src.prompts import CONFIDENCE_PROMPT
from backend.src.prompts import VALIDATION_PROMPT
from backend.src.prompts.trigger_reason_prompt import TRIGGER_REASON_PROMPT

logger = logging.getLogger(__name__)


class EmailClassifierProcessor:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION
        )

        self.base_agent = BaseAgent()
        self.model_name = AZURE_OPENAI_DEPLOYMENT_NAME
        self.max_input_tokens = MAX_INPUT_TOKEN

        self.email_classifier_agent = self.base_agent.create_agent(
            name=EMAIL_CLASSIFIER_AGENT_NAME,
            prompt=CLASSIFICATION_PROMPT
        )
        self.validator_agent = self.base_agent.create_agent(
            name=VALIDATOR_AGENT_NAME,
            prompt=VALIDATION_PROMPT
        )
        self.confidence_agent = self.base_agent.create_agent(
            name=CONFIDENCE_AGENT_NAME,
            prompt=CONFIDENCE_PROMPT
        )
        self.trigger_reason_agent = self.base_agent.create_agent(
            name=TIGGER_REASON_AGENT_NAME,
            prompt=TRIGGER_REASON_PROMPT
        )
        self.classification_with_llm = self.base_agent.create_agent(
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

    async def classify_email(self, message: str) -> str:
        result = await self.email_classifier_agent.run(task=message)
        return list(result.messages)[-1].content

    async def validate_classification(self, subject: str, body: str, classification: str) -> str:
        message = f"Email Subject: {subject}\nEmail Body: {body}\nClassification Given: {classification}"
        result = await self.validator_agent.run(task=message)
        return list(result.messages)[-1].content

    async def calculate_confidence(self, subject: str, body: str, classification: str) -> float:
        message = f"Email Subject: {subject}\nEmail Body: {body}\nClassification Given: {classification}"
        result = await self.confidence_agent.run(task=message)

        try:
            return float(json.loads(list(result.messages)[-1].content).get("confidence", 0.0))
        except Exception:
            return 0.0

    async def get_trigger_reason(
        self,
        subject: str,
        body: str,
        classification: str,
        confidence: float,
        validation: str
    ) -> str:
        message = f"""Subject: {subject}
Body: {body}
Classification: {classification}
Confidence: {confidence}
Validation Result: {validation}"""

        try:
            result = await self.trigger_reason_agent.run(task=message)
            return json.loads(list(result.messages)[-1].content).get("reason", "Unknown reason.")
        except Exception:
            return "Reason could not be determined due to an internal error."

    async def trigger_message(
        self,
        subject: str,
        body: str,
        classification: str,
        confidence: float,
        validation: str = "Invalid"
    ) -> str:
        reason = await self.get_trigger_reason(subject, body, classification, confidence, validation)
        return f"Message Triggered, Reason: {reason}"

    async def process_email(self, subject: str, body: str) -> str:
        combined_input = f"Subject: {subject}\n\nBody: {body}"
        truncated_input, was_truncated = self._truncate_to_max_tokens(combined_input)

        if was_truncated:
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
                logger.warning("No text was returned by the VLM model.")
                raise RuntimeError("No content extracted from image.")

            return assistant_reply

        except Exception as e:
            logger.exception("❌ OCR extraction via Azure OpenAI failed.")
            raise RuntimeError(f"OCR Extraction Failed: {e}") from e
