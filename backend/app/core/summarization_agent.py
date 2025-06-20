from backend.app.core.base_agent import BaseAgent
from backend.config.dev_config import SUMMARIZATION_AGENT_NAME
from backend.prompts.summarization_prompt import SUMMARIZATION_PROMPT


class SummarizationAgent:
    def __init__(self):
        self.base_agents = BaseAgent()

    def summarize_text(self, text: str, task_prompt: str = "") -> str:
        prompt = SUMMARIZATION_PROMPT + "\n\n" + task_prompt if task_prompt else SUMMARIZATION_PROMPT

        summarization_agent = self.base_agents.create_agent(SUMMARIZATION_AGENT_NAME, prompt)

        message = [{"role": "user", "content": str(text)}]
        response = summarization_agent.generate_reply(messages=message)

        return response.get("content", "") if isinstance(response, dict) else str(response).strip()

