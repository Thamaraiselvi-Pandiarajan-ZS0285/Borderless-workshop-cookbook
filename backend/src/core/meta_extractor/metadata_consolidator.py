import json
import logging
import re
from typing import List

from backend.src.config.dev_config import AZURE_OPENAI_DEPLOYMENT_NAME
from backend.src.core.base_client.base_client import OpenAiClient
from backend.src.prompts.meta_data_extraction import CONSOLIDATION_PROMPT

logger = logging.getLogger(__name__)

class MetadataConsolidatorAgent:
    def __init__(self):
        try:
            self.client = OpenAiClient().open_ai_client
            self.model = AZURE_OPENAI_DEPLOYMENT_NAME
        except Exception as e:
            logger.exception("Failed to initialize MetadataConsolidatorAgent.")
            raise RuntimeError("Initialization failed.") from e

    def consolidate(self, json_strings: List[str], category: str) -> dict:
        """
        Consolidates multiple JSON metadata strings into a single unified dictionary.
        Args:
            json_strings (List[str]): A list of JSON-formatted metadata strings.
            category (str): The metadata category to be consolidated (currently unused).
        Returns:
            dict: The consolidated metadata.
        Raises:
            RuntimeError: If any error occurs during the consolidation process.
        """
        user_input = self._format_input(json_strings)

        try:
            logger.debug("Sending request to OpenAI model.")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CONSOLIDATION_PROMPT},
                    {"role": "user", "content": user_input}
                ],
                temperature=0
            )
            result = response.choices[0].message.content.strip()
            cleaned_result = self._clean_json_response(result)
            return json.loads(cleaned_result)

        except json.JSONDecodeError as jde:
            logger.exception("Failed to decode JSON from OpenAI response.")
            raise RuntimeError("Invalid JSON returned from OpenAI model.") from jde

        except Exception as e:
            logger.exception("Consolidation failed due to an unexpected error.")
            raise RuntimeError(f"Consolidation failed: {e}") from e

    def _format_input(self, json_strings: List[str]) -> str:
        """Formats the list of JSON strings into a prompt-friendly format."""
        return "\n".join([f"Metadata Part {i+1}:\n{js}" for i, js in enumerate(json_strings)])

    def _clean_json_response(self, response: str) -> str:
        """Cleans triple-backtick and json markers from response string."""
        return re.sub(r"^```json\s*|\s*```$", "", response.strip())
