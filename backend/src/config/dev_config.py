from dotenv import load_dotenv
load_dotenv()


IMAGE_FORMATE = "jpeg"
IMAGE_RESOLUTION = 500
MAX_INPUT_TOKEN = 4096
MAX_OUTPUT_TOKEN = 4096
MAX_RETRY =5
REQUEST_TIME_OUT = 60
TEMPERATURE = 0.0
AZURE_API_TYPE = "azure"
EMAIL_CLASSIFIER_AGENT_NAME="EmailClassifierAgent"
VALIDATOR_AGENT_NAME = "ValidatorAgent"
CONFIDENCE_AGENT_NAME ="ConfidenceAgent"
TIGGER_REASON_AGENT_NAME ="TriggerReasonAgent"
VLM_CLASSIFICATION_AGENT_NAME="ClassificationAgent"
SUMMARIZATION_AGENT_NAME="SummarizationAgent"
USER_QUERY_AGENT_NAME="UserQueryAgent"
MODEL_INFO = {
    "family": "gpt-4",
    "vision": False,
    "json_output": False,
    "function_calling": False
}

AZURE_OPENAI_API_KEY="2K3oZXs28WZE2y1Fzg1jIPUdPGSY3xF0cWPcDx2DlF4RpUKimG0DJQQJ99BFACYeBjFXJ3w3AAABACOG9Osf"
AZURE_OPENAI_ENDPOINT= "https://bap-openai.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-41-mini"
AZURE_OPENAI_API_VERSION="2025-01-01-preview"
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1-mini"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT ="text-embedding-3-small"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME ="text-embedding-3-small"
AZURE_OPENAI_MODEL_NAME ="gpt-4.1-mini-2025-04-14"
DEFAULT_IMAGE_FORMAT = ".jpg"
EMAIL_TO_PDF_PATH = "/data"

BASE_PATH = "/home/dinesh.krishna@zucisystems.com/workspace/data/"
