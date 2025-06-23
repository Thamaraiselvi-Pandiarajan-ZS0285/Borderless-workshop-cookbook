import os
import json

import numpy as np
from openai import AzureOpenAI
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import sessionmaker

from backend.config.dev_config import AZURE_OPENAI_DEPLOYMENT_NAME
from backend.models.all_db_models import  EmailContentEmbedding, MetadataExtractionJsonEmbedding
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from backend.prompts.user_query_prompt import USER_QUERY_SYSTEM_PROMPT

load_dotenv()


class Embedder:
    def __init__(self, db_engine:Engine,db_session:sessionmaker):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        )
        self.db_engine = db_engine
        self.db_session = db_session

    def embed_text(self, text):
        if not isinstance(text, str):
            raise TypeError(f"Expected string input for embedding, got {type(text)}")

        if not text.strip():
            raise ValueError("Cannot embed empty text.")

        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def ingest_email_for_content(self, email_id, text, emb):
            # emb = self.embed_text(text)
            record = EmailContentEmbedding(email_id=email_id, email_content=text, embedding=emb)
            with self.db_session() as session:
                session.add(record)
                session.commit()
                session.refresh(record)

    def ingest_email_metadata_json(self, email_id, text, emb):
            # emb = self.embed_text(text)
            record = MetadataExtractionJsonEmbedding(email_id=email_id, json_content=text, embedding=emb)
            with self.db_session() as session:
                session.add(record)
                session.commit()
                session.refresh(record)

    def minify_json(self,json_input):
        try:
            if isinstance(json_input, str):
                parsed = json.loads(json_input)
            else:
                parsed = json_input
            return json.dumps(parsed, separators=(',', ':'), ensure_ascii=False)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON input: {e}")
            return None

    # def cosine_similarity(self, vec1, vec2):
    #     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    #
    # def semantic_search(self, query_embedding, top_k=10):
    #     with self.db_session() as session:
    #         records = session.query(MetadataExtractionJsonEmbedding).all()
    #
    #         # Compute similarity for each stored embedding
    #         results = []
    #         for record in records:
    #             sim = self.cosine_similarity(query_embedding, record.embedding)
    #             results.append((sim, record))
    #
    #         # Sort by similarity descending
    #         top_matches = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
    #         return [(sim, rec.json_content) for sim, rec in top_matches]
    #
    #
    # def rerank_with_cross_encoder(self, query, candidate_texts):
    #     cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    #     pairs = [[query, text] for text in candidate_texts]
    #     scores = cross_encoder.predict(pairs)
    #
    #     reranked = sorted(zip(scores, candidate_texts), key=lambda x: x[0], reverse=True)
    #     return reranked
    #
    # def format_reranked_results(self, reranked):
    #     return "\n\n".join([f"Context {i + 1}:\n{text}" for i, (_, text) in enumerate(reranked)])
    #
    # def answer_query(self, query_input, formatted_reranked_results):
    #     user_prompt = (
    #     f"User Query: {query_input}\n\n"
    #     f"Top Matching Content:\n{formatted_reranked_results}"
    # )
    #     response = self.client.chat.completions.create(
    #         model=AZURE_OPENAI_DEPLOYMENT_NAME,
    #         messages=[
    #             {"role": "system", "content":  USER_QUERY_SYSTEM_PROMPT},
    #             {"role": "user", "content": user_prompt}
    #         ],
    #         temperature=0
    #     )
    #     response = response.choices[0].message.content
    #     return response