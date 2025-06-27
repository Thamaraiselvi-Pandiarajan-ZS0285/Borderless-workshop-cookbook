from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from backend.config.dev_config import *


class BaseAgent:
    def __init__(self):
        self.model_client = AzureOpenAIChatCompletionClient(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            api_key=AZURE_OPENAI_API_KEY,
            model_info=MODEL_INFO
        )

    def create_agent(self, name: str, prompt: str) -> AssistantAgent:
        return AssistantAgent(
            name=name,
            system_message=prompt,
            model_client=self.model_client
        )
