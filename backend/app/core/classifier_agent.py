import json
import tiktoken
import logging
from backend.app.core.base_agent import BaseAgent
from backend.config.dev_config import MAX_INPUT_TOKEN, AZURE_OPENAI_DEPLOYMENT_NAME, EMAIL_CLASSIFIER_AGENT_NAME, \
    VALIDATOR_AGENT_NAME, CONFIDENCE_AGENT_NAME, TIGGER_REASON_AGENT_NAME
from backend.prompts.confidence_prompt import CONFIDENCE_PROMPT
from backend.prompts.validator_prompt import VALIDATION_PROMPT
from backend.prompts.emailclassifer_prompt import CLASSIFICATION_PROMPT
from backend.prompts.trigger_reason_prompt import TRIGGER_REASON_PROMPT

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EmailClassifierProcessor:
    """
    Processes an email by classifying it, validating the classification,
    estimating confidence, and determining if it should be flagged with a trigger message.
    """

    def __init__(self):
        self.base_agents = BaseAgent()
        self.model_name = AZURE_OPENAI_DEPLOYMENT_NAME
        self.max_input_tokens = MAX_INPUT_TOKEN

        # Initialize all agents with respective prompts
        self.email_classifier_agent = self.base_agents.create_agent(EMAIL_CLASSIFIER_AGENT_NAME, CLASSIFICATION_PROMPT)
        self.validator_agent = self.base_agents.create_agent(VALIDATOR_AGENT_NAME, VALIDATION_PROMPT)
        self.confidence_agent = self.base_agents.create_agent(CONFIDENCE_AGENT_NAME, CONFIDENCE_PROMPT)
        self.trigger_reason_agent = self.base_agents.create_agent(TIGGER_REASON_AGENT_NAME, TRIGGER_REASON_PROMPT)

    def _truncate_to_max_tokens(self, text: str) -> tuple[str, bool]:
        """
        Truncate text to stay within the token limit.

        Returns:
            (truncated_text, was_truncated)
        """
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
        """
        Classify the email using the classification agent.
        """
        logger.info("Classifying email content.")
        return self.email_classifier_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        ).strip()

    def validate_classification(self, subject: str, body: str, classification: str) -> str:
        """
        Validate the classification using the validator agent.
        """
        logger.info("Validating email classification.")
        message = f"Email Subject: {subject}\nEmail Body: {body}\nClassification Given: {classification}"
        return self.validator_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        ).strip()

    def calculate_confidence(self, subject: str, body: str, classification: str) -> float:
        """
        Calculate the confidence score for the classification.
        """
        logger.info("Calculating confidence score.")
        message = f"Email Subject: {subject}\nEmail Body: {body}\nClassification Given: {classification}"
        raw = self.confidence_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        ).strip()

        try:
            return float(json.loads(raw).get("confidence", 0.0))
        except Exception as e:
            logger.warning(f"Failed to parse confidence response: {e}")
            return 0.0

    def get_trigger_reason(self, subject: str, body: str, classification: str, confidence: float,
                           validation: str) -> str:
        """
        Determine the trigger reason based on classification, confidence, and validation.
        """
        logger.info("Determining trigger reason.")
        message = f"""Subject: {subject}
Body: {body}
Classification: {classification}
Confidence: {confidence}
Validation Result: {validation}
"""
        try:
            raw = self.trigger_reason_agent.generate_reply(
                messages=[{"role": "user", "content": message}]
            )
            return json.loads(raw.strip()).get("reason", "Unknown reason.")
        except Exception as e:
            logger.warning(f"Error determining trigger reason: {e}")
            return "Reason could not be determined due to an internal error."

    def trigger_message(self, subject: str, body: str, classification: str, confidence: float,
                        validation: str = "Invalid") -> str:
        """
        Return a formatted message for triggered events.
        """
        reason = self.get_trigger_reason(subject, body, classification, confidence, validation)
        trigger_message_status = f"Message Triggered, Reason: {reason}"
        logger.info(trigger_message_status)
        return trigger_message_status

    def process_email(self, subject: str, body: str) -> str:
        """
        Complete classification flow:
        - Truncate input if needed
        - Classify
        - Validate
        - Calculate confidence
        - Trigger Message if needed
        """
        logger.info("Starting email processing.")
        combined_input = f"Subject: {subject}\n\nBody: {body}"
        truncated_input, was_truncated = self._truncate_to_max_tokens(combined_input)

        if was_truncated:
            logger.warning("Input was truncated due to token limits.")
            return self.trigger_message(subject, body, "Unclear", 0.0, "Invalid")

        classification = self.classify_email(truncated_input)
        validation = self.validate_classification(subject, body, classification)
        confidence = self.calculate_confidence(subject, body, classification)

        if validation == "Invalid" and confidence < 95 and classification.lower() == "unclear":
            return self.trigger_message(subject, body, classification, confidence, validation)

        logger.info(f"Final Classification: {classification} | Confidence: {confidence} | Validation: {validation}")
        return classification
