from langchain_core.prompts import PromptTemplate
from openai import AzureOpenAI
from backend.prompts.reranker_prompt import RERANK_PROMPT

from backend.config.dev_config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT
)

class Reranker:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        reranker_prompt = PromptTemplate(
            template=RERANK_PROMPT,
            input_variables=["query", "passages"]
        )

        user_prompt = reranker_prompt.format(query=query, passages = results)

        print(user_prompt)

        response = self.client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": user_prompt}]
        )
        # Simulate score assignment
        ranked = sorted(results, key=lambda x: x['score'], reverse=True)
        return ranked
