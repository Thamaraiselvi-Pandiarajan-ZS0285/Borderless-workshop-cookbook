import json
import logging
from backend.src.core.base_agents.base_agent import BaseAgent
from backend.src.core.base_client.base_client import OpenAiClient
from backend.src.prompts.meta_data_extraction import VALIDATION_PROMPT_TEMPLATE

logging.basicConfig(
    level=logging.DEBUG,  # Use INFO or WARNING in production
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class MetadataValidatorAgent:
    """
    A class that validates metadata against a category-specific schema or rules using an LLM.
    """

    def __init__(self):
        try:
            self.client = OpenAiClient().open_ai_chat_completion_client
            self.base_agent = BaseAgent(self.client)
            self.validator_agent = self.base_agent.create_assistant_agent(
                "VALIDATOR_AGENT",
                "You are a metadata validation assistant."
            )
        except Exception as e:
            logging.exception("Failed to initialize MetadataValidatorAgent.")
            raise RuntimeError("Initialization of MetadataValidatorAgent failed.") from e

    async def validate_metadata(self, metadata: dict, category: str) -> dict:
        """
        Validates a given metadata dictionary using a validation prompt.

        Args:
            metadata (dict): The metadata dictionary to validate.
            category (str): The category for which the metadata should be validated.

        Returns:
            dict: A dictionary representing the validation results.

        Raises:
            RuntimeError: If validation fails or the model response is invalid.
        """
        metadata_json = json.dumps(metadata, indent=2)
        prompt = VALIDATION_PROMPT_TEMPLATE.format(category=category, metadata_json=metadata_json)

        try:
            logging.debug("Sending metadata validation task to LLM.")
            response = await self.validator_agent.run(task=prompt)
            validation_response = response.choices[0].message.content

            logging.debug("Received response from validator agent.")
            validation_result = json.loads(validation_response)
            return validation_result

        except json.JSONDecodeError as jde:
            logging.exception("Failed to parse JSON from validation response.")
            raise RuntimeError("Validator returned invalid JSON.") from jde

        except Exception as e:
            logging.exception("Metadata validation failed due to an unexpected error.")
            raise RuntimeError(f"Metadata validation failed: {e}") from e
