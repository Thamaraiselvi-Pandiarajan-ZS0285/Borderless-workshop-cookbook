from autogen import AssistantAgent
from backend.config.dev_config import *
from backend.prompts.summarization_prompt import SUMMARIZATION_PROMPT
import os
from dotenv import load_dotenv


class SummarizationAgent:
    def __init__(self):
        load_dotenv()

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



    def _create_agent(self, name: str, prompt: str) -> AssistantAgent:
        return AssistantAgent(
            name=name,
            system_message=prompt,
            llm_config=self.llm_config
        )

    def summarize_text(self, text: str, task_prompt: str = "") -> str:
        prompt = SUMMARIZATION_PROMPT + "\n\n" + task_prompt if task_prompt else SUMMARIZATION_PROMPT

        summarization_agent = self._create_agent(SUMMARIZATION_AGENT_NAME, prompt)

        message = [{"role": "user", "content": str(text)}]
        response = summarization_agent.generate_reply(messages=message)

        return response.get("content", "") if isinstance(response, dict) else str(response).strip()

