import json
from openai import AzureOpenAI
from backend.config.dev_config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_DEPLOYMENT_NAME
)
from backend.prompts.meta_data_extraction import VALIDATION_PROMPT_TEMPLATE


class MetadataValidatorAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION
        )
        self.model = AZURE_OPENAI_DEPLOYMENT_NAME

    def validate_metadata(self, metadata: dict, category: str) -> dict:
        metadata_json = json.dumps(metadata, indent=2)
        prompt = VALIDATION_PROMPT_TEMPLATE.format(category=category, metadata_json=metadata_json)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a metadata validation assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            validation_response = response.choices[0].message.content

            # Parse JSON output from the model
            validation_result = json.loads(validation_response)
            return validation_result

        except Exception as e:
            raise RuntimeError(f"Metadata validation failed: {e}")
