from backend.app.core.query_decomposer import QueryDecomposer
from backend.app.core.retriever import Retriever
from openai import AzureOpenAI
from sqlalchemy.orm import sessionmaker

from backend.config.dev_config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT
)

class RetrievalAgent:
    def __init__(self, db_session:sessionmaker):
        self.decomposer = QueryDecomposer()
        self.retriever = Retriever(db_session)
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

    def answer(self, query: str) -> str:
        print(query)
        subqueries = self.decomposer.decompose(query)
        full_context = ""
        for sq in subqueries:
            context = self.retriever.retrieve(sq)
            full_context += f"\n\n### Context for: {sq}\n{context}"

        final_prompt = f"""Using the following context, answer the original query:
Original Query: {query}
Context: {full_context}
Answer:"""

        response = self.client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": final_prompt}]
        )
        return response.choices[0].message.content.strip()
