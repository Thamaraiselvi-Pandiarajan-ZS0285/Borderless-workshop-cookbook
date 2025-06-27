import logging

from backend.app.core.base_agent import BaseAgent
from backend.src.prompts import SEMANTIC_DECOMPOSITION_PROMPT

logger = logging.getLogger(__name__)

class UserQueryAgent:
    def __init__(self) -> None:
       self.base_agent=BaseAgent()
       self.query_decomposition_agent = self.base_agent.create_agent("QUERY_DECOMPOSITION_AGENT",SEMANTIC_DECOMPOSITION_PROMPT)

    async def query_decomposition(self, user_query: str) -> str:
        try:
            result = await self.query_decomposition_agent.run(task=user_query)
            return result.messages[-1].content
        except Exception as e:
            logger.exception("Failed to decompose query.")
            raise RuntimeError(f"Query Decomposition failed: {e}")
