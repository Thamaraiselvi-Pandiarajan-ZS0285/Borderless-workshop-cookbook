from langchain_core.prompts import PromptTemplate
from openai import AzureOpenAI
from backend.prompts.query_decomposer_prompt import QUERY_DECOMPOSE_PROMPT

from backend.config.dev_config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT
)

class QueryDecomposer:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

    def decompose(self, user_query: str) -> list[str]:
        decomposer_prompt = PromptTemplate(
            template=QUERY_DECOMPOSE_PROMPT,
            input_variables=["query"]
        )

        user_prompt = decomposer_prompt.format(query=user_query)

        response = self.client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": user_prompt}
                    ]
        )
        resp = response.choices[0].message.content.strip().split("\n")
        print(resp)
        return [s.strip("- ") for s in resp]
