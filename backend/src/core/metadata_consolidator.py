import json
import re
from typing import List
from openai import AzureOpenAI
from backend.src.prompts import CONSOLIDATION_PROMPT



class MetadataConsolidatorAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION
        )
        self.model = AZURE_OPENAI_DEPLOYMENT_NAME

    def consolidate(self, json_strings: List[str], category: str) -> dict:
        user_input = "\n".join([f"Metadata Part {i+1}:\n{js}" for i, js in enumerate(json_strings)])
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CONSOLIDATION_PROMPT},
                    {"role": "user", "content": user_input}
                ],
                temperature=0
            )
            result = response.choices[0].message.content.strip()
            cleaned = re.sub(r"^```json\s*|\s*```$", "", result)
            return json.loads(cleaned)

        except Exception as e:
            raise RuntimeError(f"Consolidation failed: {e}")