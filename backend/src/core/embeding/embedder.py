import json
import logging
import numpy as np
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import sessionmaker

from backend.src.config.dev_config import USER_QUERY_AGENT_NAME
from backend.src.core.base_agents.base_agent import BaseAgent
from backend.src.core.base_client.base_client import OpenAiClient
from backend.src.db.models.email_content_embedding import EmailContentEmbedding
from backend.src.db.models.metadata_extraction_json_embedding import MetadataExtractionJsonEmbedding
from sentence_transformers import CrossEncoder

from backend.src.prompts.user_query_prompt import USER_QUERY_SYSTEM_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Embedder:
    """
    Handles embedding, ingestion, semantic search, reranking, and question answering
    on email content and metadata using OpenAI and Sentence Transformers.
    """

    def __init__(self, db_engine: Engine, db_session: sessionmaker):
        """
        Initialize the Embedder with a database engine and session.
        """
        try:
            self.client = OpenAiClient().open_ai_chat_completion_client
            self.base_agent = BaseAgent(self.client)
            self.db_engine = db_engine
            self.db_session = db_session
            self.user_query_agent = self.base_agent.create_assistant_agent(
                name=USER_QUERY_AGENT_NAME,
                prompt=USER_QUERY_SYSTEM_PROMPT
            )
        except Exception as e:
            logger.exception("Error initializing Embedder: %s", str(e))
            raise

    def embed_text(self, text: str) -> list:
        """
        Generate embedding for a given text using OpenAI's embedding model.

        :param text: Text to embed
        :return: Embedding vector
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected string input for embedding, got {type(text)}")

        if not text.strip():
            raise ValueError("Cannot embed empty text.")

        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.exception("Failed to embed text: %s", str(e))
            raise

    def ingest_email_for_content(self, email_id: str, text: str, emb: list) -> None:
        """
        Save email content and its embedding to the database.

        :param email_id: Unique ID of the email
        :param text: Email content
        :param emb: Embedding vector
        """
        try:
            record = EmailContentEmbedding(email_id=email_id, email_content=text, embedding=emb)
            with self.db_session() as session:
                session.add(record)
                session.commit()
                session.refresh(record)
        except Exception as e:
            logger.exception("Failed to ingest email content: %s", str(e))
            raise

    def ingest_email_metadata_json(self, email_id: str, text: str, emb: list) -> None:
        """
        Save email metadata in JSON format and its embedding to the database.

        :param email_id: Unique ID of the email
        :param text: JSON string of metadata
        :param emb: Embedding vector
        """
        try:
            record = MetadataExtractionJsonEmbedding(email_id=email_id, json_content=text, embedding=emb)
            with self.db_session() as session:
                session.add(record)
                session.commit()
                session.refresh(record)
        except Exception as e:
            logger.exception("Failed to ingest email metadata JSON: %s", str(e))
            raise

    def minify_json(self, json_input) -> str:
        """
        Minify JSON by removing unnecessary whitespace.

        :param json_input: JSON string or dict
        :return: Minified JSON string
        """
        try:
            parsed = json.loads(json_input) if isinstance(json_input, str) else json_input
            return json.dumps(parsed, separators=(',', ':'), ensure_ascii=False)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON input: %s", str(e))
            return None

    def cosine_similarity(self, vec1: list, vec2: list) -> float:
        """
        Compute cosine similarity between two vectors.

        :param vec1: First vector
        :param vec2: Second vector
        :return: Cosine similarity score
        """
        try:
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except Exception as e:
            logger.exception("Cosine similarity computation failed: %s", str(e))
            raise

    def semantic_search(self, query_embedding: list, top_k: int = 10):
        """
        Perform semantic search on stored metadata using cosine similarity.

        :param query_embedding: Query embedding vector
        :param top_k: Number of top results to return
        :return: List of tuples (similarity_score, json_content)
        """
        try:
            with self.db_session() as session:
                records = session.query(MetadataExtractionJsonEmbedding).all()

                results = []
                for record in records:
                    sim = self.cosine_similarity(query_embedding, record.embedding)
                    results.append((sim, record))

                top_matches = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
                return [(sim, rec.json_content) for sim, rec in top_matches]
        except Exception as e:
            logger.exception("Semantic search failed: %s", str(e))
            raise

    def rerank_with_cross_encoder(self, query: str, candidate_texts: list) -> list:
        """
        Rerank candidate texts using a cross-encoder.

        :param query: User query
        :param candidate_texts: List of candidate text strings
        :return: Reranked list of (score, text)
        """
        try:
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [[query, text] for text in candidate_texts]
            scores = cross_encoder.predict(pairs)

            reranked = sorted(zip(scores, candidate_texts), key=lambda x: x[0], reverse=True)
            return reranked
        except Exception as e:
            logger.exception("Cross-encoder reranking failed: %s", str(e))
            raise

    def format_reranked_results(self, reranked: list) -> str:
        """
        Format reranked results for display or use in answering queries.

        :param reranked: List of (score, text) tuples
        :return: Formatted string
        """
        return "\n\n".join([f"Context {i + 1}:\n{text}" for i, (_, text) in enumerate(reranked)])

    async def answer_query(self, query_input: str, formatted_reranked_results: str) -> str:
        """
        Generate an answer to a user query using a custom assistant agent.

        :param query_input: User question
        :param formatted_reranked_results: Texts relevant to the question
        :return: Assistant's response
        """
        try:
            user_prompt = (
                f"User Query: {query_input}\n\n"
                f"Top Matching Content:\n{formatted_reranked_results}"
            )

            response = await self.user_query_agent.run(task=user_prompt)
            content = response.choices[0].message.content.strip()
            return content
        except Exception as e:
            logger.exception("Failed to answer query: %s", str(e))
            raise
