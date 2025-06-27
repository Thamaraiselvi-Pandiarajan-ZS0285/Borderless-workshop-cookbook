from backend.src.core.base_agents.base_agent import BaseAgent
from backend.src.prompts import SUMMARIZATION_PROMPT



class SummarizationAgent:
    def __init__(self):
            self.base_agent= BaseAgent()
    async def summarize_text(self, text: str, task_prompt: str = "") -> str:
        prompt = SUMMARIZATION_PROMPT + "\n\n" + task_prompt if task_prompt else SUMMARIZATION_PROMPT

        summarization_agent = self.base_agent.create_assistant_agent(name=SUMMARIZATION_AGENT_NAME, prompt=prompt)
        response =  await summarization_agent.run(task=text)
        return response.get("content", "") if isinstance(response, dict) else str(response).strip()

