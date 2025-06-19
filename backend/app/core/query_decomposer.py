from openai import AzureOpenAI

from backend.config.dev_config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT
)

class QueryDecomposer:
    def __init__(self):
        print(AZURE_OPENAI_API_KEY)
        print(AZURE_OPENAI_API_VERSION)
        print(AZURE_OPENAI_ENDPOINT)
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

    def decompose(self, user_query: str) -> list[str]:
        prompt = f"Break down the user query into minimal sub-queries: {user_query}"
        response = self.client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}]
        )
        resp = response.choices[0].message.content.strip().split("\n\n")[1]
        print(resp)
        return [s.strip("- ") for s in resp.split("\n")]
