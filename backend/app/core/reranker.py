from openai import AzureOpenAI

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
        prompt = f"""
You are given the following query: "{query}"
And the following retrieved results:\n{[r['content'] for r in results]}
Rank these by relevance and return them with scores.
"""
        response = self.client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}]
        )
        # Simulate score assignment
        ranked = sorted(results, key=lambda x: x['score'], reverse=True)
        return ranked
