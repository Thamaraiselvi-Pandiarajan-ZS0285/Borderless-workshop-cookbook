from autogen import AssistantAgent
from backend.config.dev_config import TEMPERATURE, AZURE_API_TYPE, SUMMARIZATION_AGENT_NAME
from backend.prompts.summarization_prompt import SUMMARIZATION_PROMPT
import os
from dotenv import load_dotenv


class SummarizationAgent:
    def __init__(self):
        load_dotenv()

        self.model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        self.llm_config = {
            "config_list": [{
                "model": self.model_name,
                "api_type": AZURE_API_TYPE,
                "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "api_version": os.getenv("AZURE_OPENAI_API_VERSION")
            }],
            "temperature": TEMPERATURE,
        }

        self.summarization_agent = self._create_agent(SUMMARIZATION_AGENT_NAME, SUMMARIZATION_PROMPT)

    def _create_agent(self, name: str, prompt: str) -> AssistantAgent:
        return AssistantAgent(
            name=name,
            system_message=prompt,
            llm_config=self.llm_config
        )

    def summarize_text(self, text: str) -> str:
        message= [{"role": "user", "content": str(text)}]
        response = self.summarization_agent.generate_reply(
            messages=message
        )
        return response
