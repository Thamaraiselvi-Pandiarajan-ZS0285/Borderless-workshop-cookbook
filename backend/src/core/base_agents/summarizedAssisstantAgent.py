from autogen_agentchat.agents import AssistantAgent

from backend.src.core.base_agents.summarize_agent import SummarizeAgent


class SummarizedAssistantAgent(AssistantAgent):
    def __init__(self, summarizer: 'SummarizeAgent', **kwargs):
        super().__init__(**kwargs)
        self._summarizer = summarizer

    async def generate_reply(self, messages: str, **kwargs):
        original_reply = await super().generate_reply(messages, **kwargs)
        summarized_reply = await self._summarizer.summarize(original_reply)
        return summarized_reply
