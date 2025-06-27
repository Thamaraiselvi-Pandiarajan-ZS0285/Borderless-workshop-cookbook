from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from backend.src.config.dev_config import AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_API_KEY, MODEL_INFO

class BaseAgent:
    def __init__(self):
        self.model_client = AzureOpenAIChatCompletionClient(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            api_key=AZURE_OPENAI_API_KEY,
            model_info=MODEL_INFO
        )
