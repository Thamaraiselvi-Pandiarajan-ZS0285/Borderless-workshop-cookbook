from autogen import AssistantAgent

from backend.config.dev_config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    TEMPERATURE,
    AZURE_API_TYPE
)
class BaseAgent:
    def __init__(self):

        self.model_name = AZURE_OPENAI_DEPLOYMENT_NAME

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

    def create_agent(self, name: str, prompt: str) -> AssistantAgent:
        return AssistantAgent(
            name=name,
            system_message=prompt,
            llm_config=self.llm_config
        )