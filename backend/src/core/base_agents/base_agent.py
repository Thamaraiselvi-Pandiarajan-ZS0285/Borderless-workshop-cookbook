from autogen_agentchat.agents import AssistantAgent

class BaseAgent:
    def __init__(self, model_client):
        self.model_client = model_client

    def create_assistant_agent(self, name: str, prompt: str) -> AssistantAgent:
        return AssistantAgent(
            name=name,
            system_message=prompt,
            model_client=self.model_client
        )