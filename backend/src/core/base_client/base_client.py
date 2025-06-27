from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from openai import AzureOpenAI

from backend.src.config.dev_config import AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_API_KEY, MODEL_INFO

class OpenAiClient:
    def __init__(self):
        self.open_ai_chat_completion_client = None
        self.open_ai_client = None

    def initiate_open_ai_chat_completion_client(self):
        self.open_ai_chat_completion_client = AzureOpenAIChatCompletionClient(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            api_key=AZURE_OPENAI_API_KEY,
            model_info=MODEL_INFO
        )
    def initiate_open_ai_client(self):
        self.open_ai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION
        )