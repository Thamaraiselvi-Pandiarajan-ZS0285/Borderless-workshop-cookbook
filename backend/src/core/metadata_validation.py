import json
from backend.src.core.base_agents.base_agent import BaseAgent
from backend.src.prompts import VALIDATION_PROMPT_TEMPLATE


class MetadataValidatorAgent:
    def __init__(self):
        self.base_agent=BaseAgent()


async def validate_metadata(self, metadata: dict, category: str) -> dict:
        metadata_json = json.dumps(metadata, indent=2)
        prompt = VALIDATION_PROMPT_TEMPLATE.format(category=category, metadata_json=metadata_json)
        self.validator_agent = self.base_agent.create_agent("VALIDATOR_AGENT", "You are a metadata validation assistant.")
        try:
            response = await self.validator_agent.run(task=prompt)
            validation_response = response.choices[0].message.content

            # Parse JSON output from the model
            validation_result = json.loads(validation_response)
            return validation_result

        except Exception as e:
            raise RuntimeError(f"Metadata validation failed: {e}")
