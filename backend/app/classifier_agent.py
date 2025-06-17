import json
import os
import tiktoken
from autogen import AssistantAgent
from dotenv import load_dotenv
from backend.prompts.confidence_prompt import CONFIDENCE_PROMPT
from backend.prompts.validator_prompt import VALIDATION_PROMPT
from backend.prompts.emailclassifer_prompt import CLASSIFICATION_PROMPT
from backend.prompts.trigger_reason_prompt import TRIGGER_REASON_PROMPT


load_dotenv()

llm_config = {
    "config_list": [{
        "model": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        "api_type": "azure",
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION")
    }],
    "temperature": 0.0,
}
MAX_INPUT_TOKENS = 1000
MODEL_NAME=os.getenv("AZURE_OPENAI_DEPLOYMENT")


VALIDATION_PROMPT_TEMPLATE = """You are a validation agent. Given an email's subject and body, and a classification result, determine if the classification is accurate.
Email Subject: {subject}
Email Body: {body}
Classifier Output: {classification}
"""

# Create agents
email_classifier_agent = AssistantAgent(
    name="EmailClassifierAgent",
    system_message=CLASSIFICATION_PROMPT,
    llm_config=llm_config
)

validator_agent = AssistantAgent(
    name="ValidatorAgent",
    system_message=VALIDATION_PROMPT,
    llm_config=llm_config
)

confidence_agent = AssistantAgent(
    name="ConfidenceAgent",
    system_message=CONFIDENCE_PROMPT,
    llm_config=llm_config
)

trigger_reason_agent = AssistantAgent(
    name="TriggerReasonAgent",
    system_message=TRIGGER_REASON_PROMPT,
    llm_config=llm_config
)

def truncate_to_max_tokens(text: str, max_tokens: int = MAX_INPUT_TOKENS, model: str = MODEL_NAME) -> tuple[str, bool]:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    tokens = enc.encode(text)
    was_truncated = len(tokens) > max_tokens

    if was_truncated:
        tokens = tokens[:max_tokens]

    return enc.decode(tokens), was_truncated

def classify_email(message: str) -> str:
    return email_classifier_agent.generate_reply(
        messages=[{"role": "user", "content": message}]
    ).strip()

def validate_classification(subject: str, body: str, classification: str) -> str:
    message = f"""
    Email Subject: {subject}
    Email Body: {body}
    Classification Given: {classification}
    """
    return validator_agent.generate_reply(
        messages=[{"role": "user", "content": message}]
    ).strip()

def calculate_confidence(subject: str, body: str, classification: str) -> float:
    message = f"""
    Email Subject: {subject}
    Email Body: {body}
    Classification Given: {classification}
    """
    raw = confidence_agent.generate_reply(
        messages=[{"role": "user", "content": message}]
    ).strip()

    try:
        return float(json.loads(raw)["confidence"])
    except Exception:
        return 0.0

def get_trigger_reason(subject: str, body: str, classification: str, confidence: float, validation: str) -> str:
    message = f"""
    Subject: {subject}
    Body: {body}
    Classification: {classification}
    Confidence: {confidence}
    Validation Result: {validation}
    """
    try:
        raw = trigger_reason_agent.generate_reply(messages=[{"role": "user", "content": message}])
        return json.loads(raw.strip())["reason"]
    except Exception:
        return "Reason could not be determined due to an internal error."

def trigger_message(subject: str, body: str, classification: str, confidence: float, validation: str = "Invalid") -> dict:
    reason = get_trigger_reason(subject, body, classification, confidence, validation)
    return {
        "status": "Escalation triggered",
        "reason": reason,
        "classification": classification,
        "confidence": confidence
    }

def process_email(subject: str, body: str) -> dict:
    combined_input = f"Subject: {subject}\n\nBody: {body}"
    truncated_input, was_truncated = truncate_to_max_tokens(combined_input)

    if was_truncated:
        return trigger_message(subject, body, "Unclear", 0.0, "Invalid")

    classification = classify_email(combined_input)
    validation = validate_classification(subject, body, classification)
    confidence = float(calculate_confidence(subject, body, classification))

    response = {
        "confidence": confidence,
        "classification": classification,
        "validation": validation
    }

    if validation == "Invalid" and confidence < 95 and classification.lower() == "unclear":
        response = trigger_message(subject, body, classification, confidence, validation)

    return response
