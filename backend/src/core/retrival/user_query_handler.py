import logging

from backend.src.core.base_agents.base_agent import BaseAgent
from backend.src.core.base_client.base_client import OpenAiClient
from backend.src.prompts.decomposition_prompt import SEMANTIC_DECOMPOSITION_PROMPT

logger = logging.getLogger(__name__)


class UserQueryAgent:
    """
    Handles user query processing by leveraging a semantic decomposition agent
    to break down complex queries into simpler sub-tasks using an LLM.

    Attributes:
        client: An instance of the OpenAI client.
        base_agent: A BaseAgent initialized with the client.
        query_decomposition_agent: An assistant agent configured for query decomposition.
    """

    def __init__(self) -> None:
        """
        Initializes the UserQueryAgent with necessary clients and agents.
        """
        try:
            self.client = OpenAiClient().open_ai_chat_completion_client
            self.base_agent = BaseAgent(self.client)
            self.query_decomposition_agent = self.base_agent.create_assistant_agent(
                "QUERY_DECOMPOSITION_AGENT",
                SEMANTIC_DECOMPOSITION_PROMPT
            )
            logger.info("UserQueryAgent initialized successfully.")
        except Exception as e:
            logger.exception("Failed to initialize UserQueryAgent.")
            raise RuntimeError(f"Initialization failed: {e}")

    async def query_decomposition(self, user_query: str) -> str:
        """
        Decomposes a user query into simpler tasks using the decomposition agent.

        Args:
            user_query (str): The input query from the user.

        Returns:
            str: The content of the final message from the agent's response.

        Raises:
            RuntimeError: If query decomposition fails.
        """
        if not user_query or not isinstance(user_query, str):
            logger.error("Invalid user query input: %s", user_query)
            raise ValueError("User query must be a non-empty string.")

        try:
            logger.info("Starting query decomposition for: %s", user_query)
            result = await self.query_decomposition_agent.run(task=user_query)

            if not result.messages:
                logger.error("Decomposition result has no messages.")
                raise RuntimeError("Empty response received from decomposition agent.")

            final_response = result.messages[-1].content
            logger.info("Query decomposition successful.")

            return final_response

        except Exception as e:
            logger.exception("Failed to decompose query.")
            raise RuntimeError(f"Query Decomposition failed: {e}")
