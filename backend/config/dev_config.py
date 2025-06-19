from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

IMAGE_FORMATE = "jpeg"
IMAGE_RESOLUTION = 500
MAX_INPUT_TOKEN = 1000
TEMPERATURE = 0.0
AZURE_API_TYPE = "azure"
EMAIL_CLASSIFIER_AGENT_NAME="EmailClassifierAgent"
VALIDATOR_AGENT_NAME = "ValidatorAgent"
CONFIDENCE_AGENT_NAME ="ConfidenceAgent"
TIGGER_REASON_AGENT_NAME ="TriggerReasonAgent"

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")

TENANT_ID = os.getenv("AZURE_TENANT_ID")
CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
USER_ID = os.getenv("AZURE_USER_ID")  # e.g. 'me' or actual email
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["https://graph.microsoft.com/.default"]
GRAPH_ENDPOINT = "https://graph.microsoft.com/v1.0"
MAIL_FOLDER = "Inbox"
PROCESSED_FOLDER = "Processed"
FAILED_FOLDER = "Failed"