from autogen_agentchat.agents import AssistantAgent

from backend.src.core.base_agents.summarize_agent import SummarizeAgent
from backend.src.core.base_agents.summarizedAssisstantAgent import SummarizedAssistantAgent


class BaseAgent:
    def __init__(self, model_client):
        self.model_client = model_client
        self.summarizer = SummarizeAgent(model_client)

    def create_assistant_agent(self, name: str, prompt: str) -> AssistantAgent:
        return SummarizedAssistantAgent(
            name=name,
            system_message=prompt,
            model_client=self.model_client,
            summarizer=self.summarizer
        )