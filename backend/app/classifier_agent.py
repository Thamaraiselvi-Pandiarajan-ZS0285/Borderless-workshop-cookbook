import os
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
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

# Email Classification Instruction
CLASSIFICATION_PROMPT = """You are a classification assistant trained to analyze the content and intent of business-related emails. Your task is to determine whether the email belongs to one of the following categories:
"RFP" – The email is a Request for Proposal. These emails typically invite the recipient to submit a proposal, quotation, or offer for a service or project. Keywords may include: request for proposal, quotation, pricing request, invitation to bid, scope of work.
"Winning" – The email confirms that a proposal or submission was accepted or selected. It may express congratulations, acceptance, or award. Look for phrases like: we are pleased to inform you, your proposal has been selected, you have been awarded, congratulations.
"Rejection" – The email communicates that a proposal or submission was not selected. It may include polite decline language, regrets, or suggestions for future opportunities. Common phrases: we regret to inform you, unfortunately, not selected, thank you for your submission.

Instructions:
Read the entire email content carefully.
Do not rely only on keywords—use context and tone to make an accurate decision.
Ignore irrelevant parts such as greetings, footers, or signatures unless they add meaningful context.
Respond with only one label: "RFP", "Winning", or "Rejection"—no extra words or explanations.
If the email does not clearly fit any of these categories, respond with "Unclear".
"""

VALIDATION_PROMPT_TEMPLATE = """You are a validation agent. Given an email's subject and body, and a classification result, determine if the classification is accurate.
Email Subject:{subject}
Email Body:{body}
Classifier Output:{classification}
Is this classification valid based on the content above? Reply only with 'Valid' or 'Invalid'.
"""

# Create shared agent instances
email_classifier_agent = AssistantAgent(
    name="EmailClassifierAgent",
    system_message=CLASSIFICATION_PROMPT,
    llm_config=llm_config
)

validator_agent = AssistantAgent(
    name="ValidatorAgent",
    system_message="You validate email classification accuracy. Reply only with 'Valid' or 'Invalid'.",
    llm_config=llm_config
)
def validate_classification(subject: str, body: str, classification_result: str) -> str:
    """
    Validates whether the classification result is appropriate for the email content.
    Returns 'Valid', 'Invalid', or 'Unclear'.
    """
    try:
        validation_prompt = VALIDATION_PROMPT_TEMPLATE.format(
            subject=subject,
            body=body,
            classification=classification_result
        )

        response = validator_agent.generate_reply(
            messages=[{"role": "user", "content": validation_prompt}]
        ).strip()

        return response if response in ["Valid", "Invalid"] else "Unclear"

    except Exception as e:
        return f"Validation failed: {e}"

def classify_email(subject: str, body: str) -> dict:
    """Classifies an email and validates the classification.
    Returns a dictionary with classification and validation result."""

    # User agent to initiate classification
    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False}
    )

    email_message = f"Subject: {subject}\n\nBody:\n{body}"

    try:
        classification_result = email_classifier_agent.generate_reply(
            messages=[{"role": "user", "content": email_message}]
        ).strip()

    except Exception as e:
        return {"error": f"Classification failed: {e}"}

    # call the validation method
    validation_status = validate_classification(subject, body, classification_result)

    return {
        "classification": classification_result,
        "validation": validation_status
    }
