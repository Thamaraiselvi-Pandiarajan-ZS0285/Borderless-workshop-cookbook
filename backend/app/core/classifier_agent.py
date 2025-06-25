import json
from typing import Optional

import tiktoken
from openai import AzureOpenAI
from autogen import AssistantAgent
from backend.config.dev_config import *
from backend.prompts.confidence_prompt import CONFIDENCE_PROMPT
from backend.prompts.validator_prompt import VALIDATION_PROMPT
from backend.prompts.emailclassifer_prompt import *
from backend.prompts.trigger_reason_prompt import TRIGGER_REASON_PROMPT
import logging

logger = logging.getLogger(__name__)

class EmailClassifierProcessor:
    def __init__(self):
        load_dotenv()

        self.model_name = AZURE_OPENAI_DEPLOYMENT_NAME
        self.max_input_tokens = MAX_INPUT_TOKEN

        self.llm_config = {
            "config_list": [{
                "model": self.model_name,
                "api_type": AZURE_API_TYPE,
                "api_key": AZURE_OPENAI_API_KEY,
                "base_url": AZURE_OPENAI_ENDPOINT,
                "api_version": AZURE_OPENAI_API_VERSION
            }],
            "temperature": TEMPERATURE,
        }

        try:
            self.client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_API_VERSION
            )
            self.model = AZURE_OPENAI_DEPLOYMENT_NAME
            logger.info("✅ Azure OpenAI client initialized successfully.")
        except Exception as e:
            logger.exception("❌ Failed to initialize Azure OpenAI client.")
            raise RuntimeError(f"Initialization Failed: {e}") from e


        self.email_classifier_agent = self._create_agent(EMAIL_CLASSIFIER_AGENT_NAME, CLASSIFICATION_PROMPT)
        self.validator_agent = self._create_agent(VALIDATOR_AGENT_NAME, VALIDATION_PROMPT)
        self.confidence_agent = self._create_agent(CONFIDENCE_AGENT_NAME, CONFIDENCE_PROMPT)
        self.trigger_reason_agent = self._create_agent(TIGGER_REASON_AGENT_NAME, TRIGGER_REASON_PROMPT)

    def _create_agent(self, name: str, prompt: str) -> AssistantAgent:
        return AssistantAgent(
            name=name,
            system_message=prompt,
            llm_config=self.llm_config
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

    def classify_email(self, message: str) -> str:
        return self.email_classifier_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        ).strip()

    def validate_classification(self, subject: str, body: str, classification: str) -> str:
        message = f"Email Subject: {subject}\nEmail Body: {body}\nClassification Given: {classification}"
        return self.validator_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        ).strip()

    def calculate_confidence(self, subject: str, body: str, classification: str) -> float:
        message = f"Email Subject: {subject}\nEmail Body: {body}\nClassification Given: {classification}"
        raw = self.confidence_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        ).strip()

        try:
            return float(json.loads(raw).get("confidence", 0.0))
        except Exception:
            return 0.0

    def get_trigger_reason(self, subject: str, body: str, classification: str, confidence: float,
                           validation: str) -> str:
        message = f"""Subject: {subject}
                Body: {body}
                Classification: {classification}
                Confidence: {confidence}
                Validation Result: {validation}
                """
        try:
            raw = self.trigger_reason_agent.generate_reply(messages=[{"role": "user", "content": message}])
            return json.loads(raw.strip()).get("reason", "Unknown reason.")
        except Exception:
            return "Reason could not be determined due to an internal error."

    def trigger_message(self, subject: str, body: str, classification: str, confidence: float,
                        validation: str = "Invalid") -> str:
        reason = self.get_trigger_reason(subject, body, classification, confidence, validation)
        trigger_message_status=f"Message Triggered,Reason {reason}"
        return trigger_message_status


    def process_email(self, subject: str, body: str) -> str:
        combined_input = f"Subject: {subject}\n\nBody: {body}"
        truncated_input, was_truncated = self._truncate_to_max_tokens(combined_input)

        if was_truncated:
            return self.trigger_message(subject, body, "Unclear", 0.0, "Invalid")

        classification = self.classify_email(truncated_input)
        validation = self.validate_classification(subject, body, classification)
        confidence = self.calculate_confidence(subject, body, classification)

        if validation == "Invalid" and confidence < 95 and classification.lower() == "unclear":
            return self.trigger_message(subject, body, classification, confidence, validation)

        return classification

    def classify_via_vlm(self, base64_str):
        if not base64_str or not isinstance(base64_str, str):
            raise ValueError("Input base64 string is invalid or empty.")

        prompt = CLASSIFICATION_PROMPT_VLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_str}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                temperature=0
            )

            extracted_text: Optional[str] = response.choices[0].message.content
            if not extracted_text:
                logger.warning(" No text was returned by the VLM model.")
                raise RuntimeError("No content extracted from image.")

            return extracted_text.strip()

        except Exception as e:
            logger.exception("❌ OCR extraction via Azure OpenAI failed.")
            raise RuntimeError(f"OCR Extraction Failed: {e}") from e
