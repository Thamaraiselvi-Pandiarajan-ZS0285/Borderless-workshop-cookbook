import json
import logging
from openai import AzureOpenAI

from backend.config.dev_config import *
from backend.prompts.decomposition_prompt import SEMANTIC_DECOMPOSITION_PROMPT

logger = logging.getLogger(__name__)

class UserQueryAgent:
    def __init__(self) -> None:
        try:
            self.client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_API_VERSION
            )
            self.model = AZURE_OPENAI_DEPLOYMENT_NAME
            logger.info("✅ Azure OpenAI client initialized successfully.")
        except Exception as e:
            logger.exception("❌ Failed to initialize Azure OpenAI client.")
            raise RuntimeError(f"Initialization Failed: {e}") from e

    def query_decomposition(self, user_query: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SEMANTIC_DECOMPOSITION_PROMPT},
                    {"role": "user", "content": user_query}
                ],
                temperature=0
            )
            response = response.choices[0].message.content
            return response

        except Exception as e:
            raise RuntimeError(f"Query Decomposition failed: {e}")