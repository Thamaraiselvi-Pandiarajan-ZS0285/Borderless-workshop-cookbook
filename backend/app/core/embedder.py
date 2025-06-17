import os
import openai
from backend.db.vector import sessionmaker, EmailChunk
from backend.utils.tokenizer import split_text
from sentence_transformers import CrossEncoder
from backend.utils.memory import memory

# Azure OpenAI setup
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

class Embedder:
    def __init__(self):
        self.db = sessionmaker()

    def embed_text(self, text):
        response = openai.Embedding.create(
            engine=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            input=text
        )
        return response['data'][0]['embedding']

    def ingest_email(self, email_id, text):
        chunks = split_text(text)
        for chunk in chunks:
            emb = self.embed_text(chunk)
            record = EmailChunk(email_id=email_id, chunk_text=chunk, embedding=emb)
            self.db.add(record)
        self.db.commit()

    def search(self, query, top_k=5):
        q_emb = self.embed_text(query)
        result = self.db.execute(
            f"""
            SELECT id, chunk_text, embedding <-> :q_vec AS distance
            FROM email_chunks
            ORDER BY embedding <-> :q_vec
            LIMIT :top_k
            """,
            {"q_vec": q_emb, "top_k": top_k * 4}
        ).fetchall()
        texts = [row[1] for row in result]
        ranked = cross_encoder.predict([[query, t] for t in texts])
        sorted_chunks = sorted(zip(texts, ranked), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in sorted_chunks[:top_k]]

    def respond(self, session_id, query):
        memory.add(session_id, f"User: {query}")
        top_chunks = self.search(query)
        prompt = "\n\n---\n\n".join(top_chunks + memory.get(session_id)[-5:])
        completion = openai.ChatCompletion.create(
            engine="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        reply = completion.choices[0].message['content']
        memory.add(session_id, f"Assistant: {reply}")
        return reply
