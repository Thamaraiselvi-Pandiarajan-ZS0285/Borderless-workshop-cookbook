import logging

from backend.src.config.dev_config import SUMMARIZATION_AGENT_NAME
from backend.src.core.base_agents.base_agent import BaseAgent
from backend.src.core.base_client.base_client import OpenAiClient
from backend.src.prompts.summarization_prompt import SUMMARIZATION_PROMPT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SummarizationAgent:
    """
    SummarizationAgent handles the task of summarizing input text using an AI assistant.

    Attributes:
        client: The OpenAI client for chat completion.
        base_agent: BaseAgent object to create assistant agents.
    """

    def __init__(self):
        """
        Initializes the summarization agent with the OpenAI client and base agent.
        """
        try:
            self.client = OpenAiClient()
            self.base_agent = BaseAgent(self.client.open_ai_chat_completion_client)
            logger.info("SummarizationAgent initialized successfully.")
        except Exception as e:
            logger.exception("Failed to initialize SummarizationAgent.")
            raise RuntimeError(f"Initialization failed: {e}")

    async def summarize_text(self, text: str, task_prompt: str = "") -> str:
        """
        Summarizes the given input text using an LLM agent.

        Args:
            text (str): The input text to be summarized.
            task_prompt (str, optional): An optional task-specific instruction to customize the summary.

        Returns:
            str: The summarized version of the input text.

        Raises:
            RuntimeError: If the summarization process fails.
        """
        if not text or not isinstance(text, str):
            logger.error("Invalid text input for summarization.")
            raise ValueError("Text input must be a non-empty string.")

        try:
            prompt = f"{SUMMARIZATION_PROMPT}\n\n{task_prompt}" if task_prompt else SUMMARIZATION_PROMPT
            summarization_agent = self.base_agent.create_assistant_agent(
                name=SUMMARIZATION_AGENT_NAME,
                prompt=prompt
            )

            logger.info("Summarization task started.")
            response = await summarization_agent.run(task=text)

            # Handle both dict and non-dict response formats
            if isinstance(response, dict):
                result = response.get("content", "")
            elif hasattr(response, 'messages'):  # Handle OpenAI-style response
                result = response.messages[-1].content if response.messages else ""
            else:
                result = str(response).strip()

            logger.info("Summarization completed successfully.")
            return result

        except Exception as e:
            logger.exception("Summarization failed.")
            raise RuntimeError(f"Summarization failed: {e}")
