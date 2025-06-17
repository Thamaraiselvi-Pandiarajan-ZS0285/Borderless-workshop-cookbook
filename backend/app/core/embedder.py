import os

import numpy as np
import torch
from openai import AzureOpenAI
from sqlalchemy import text
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.functions import func
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from backend.models.save_email_chunks import EmailChunk
from backend.utils.tokenizer import split_text
from dotenv import load_dotenv

load_dotenv()


#cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")


class Embedder:
    def __init__(self, db_engine:Engine,db_session:sessionmaker):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        )
        self.db_engin = db_engine
        self.db_session = db_session

    def embed_text(self, text):
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        return response.data[0].embedding

    def ingest_email(self, email_id, text):
        chunks = split_text(text)
        for chunk in chunks:
            emb = self.embed_text(chunk)
            record = EmailChunk(email_id=email_id, chunk_text=chunk, embedding=emb)
            with self.db_session() as session:
                session.add(record)
                session.commit()
                session.refresh(record)

    def semantic_search(self, query: str, limit: int = 20) -> list[dict]:
        query_emb = self.embed_text(query)
        session = self.db_session()
        try:
            results = (
                session.query(
                    EmailChunk.email_chunk_id,
                    EmailChunk.chunk_text,
                    func.l2_distance(EmailChunk.embedding, query_emb).label("distance")
                )
                .order_by("distance").limit(limit).all())
        finally:
            session.close()

        return [{"id": row[0], "text": row[1], "distance": row[2]} for row in result]

    def rerank(self, query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
        pairs = [(query, doc["text"]) for doc in docs]
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            scores = model(**inputs).logits.squeeze(-1)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1].item(), reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    def respond(self, query: str, top_k: int = 5) -> list[dict]:
        semantically_similar = self.semantic_search(query, limit=top_k * 4)
        top_chunks = self.rerank(query, semantically_similar, top_k=top_k)
        return top_chunks

