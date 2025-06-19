from openai import AzureOpenAI

from backend.config.dev_config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_MODEL
)

class EmbeddingGenerator:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.client.embeddings.create(
            model = AZURE_OPENAI_EMBEDDING_MODEL,
            input = texts
        ).data
