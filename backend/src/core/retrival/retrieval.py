import logging
from typing import List

from sqlalchemy.engine.base import Engine
from sqlalchemy.orm.session import sessionmaker

from backend.src.config.dev_config import USER_QUERY_AGENT_NAME
from backend.src.core.base_agents.base_agent import BaseAgent
from backend.src.core.base_client.base_client import OpenAiClient
from backend.src.db.models.metadata_extraction_json_embedding import MetadataExtractionJsonEmbedding
from backend.src.prompts.decomposition_prompt import SEMANTIC_DECOMPOSITION_PROMPT
import numpy as np
from sentence_transformers import CrossEncoder
from backend.src.prompts.user_query_prompt import USER_QUERY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class RetrievalInterface:
    def __init__(self,db_engine: Engine, db_session: sessionmaker):
        self.client = OpenAiClient().open_ai_chat_completion_client
        self.embedding_client = OpenAiClient().openai_embedding_client
        self.base_agent = BaseAgent(self.client)
        self.db_engine = db_engine
        self.db_session = db_session
        self.user_query_agent1 = self.base_agent.create_assistant_agent(
            name=USER_QUERY_AGENT_NAME,
            prompt=USER_QUERY_SYSTEM_PROMPT
        )
        self.query_decomposition_agent = self.base_agent.create_assistant_agent(
            "QUERY_DECOMPOSITION_AGENT",
            SEMANTIC_DECOMPOSITION_PROMPT
        )

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
            response = self.embedding_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.exception("Failed to embed text: %s", str(e))
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

    async def query_decomposition(self, user_query: str) -> str:
        """
        Decomposes a user query into simpler tasks using the decomposition agent.

        Args:
            user_query (str): The input query from the user.

        Returns:
            str: The content of the final message from the agent's response.

        Raises:
            RuntimeError: If query decomposition fails.
        """
        if not user_query or not isinstance(user_query, str):
            logger.error("Invalid user query input: %s", user_query)
            raise ValueError("User query must be a non-empty string.")

        try:
            logger.info("Starting query decomposition for: %s", user_query)
            result = await self.query_decomposition_agent.run(task=user_query)

            if not result.messages:
                logger.error("Decomposition result has no messages.")
                raise RuntimeError("Empty response received from decomposition agent.")

            final_response = result.messages[-1].content
            logger.info("Query decomposition successful.")

            return final_response

        except Exception as e:
            logger.exception("Failed to decompose query.")
            raise RuntimeError(f"Query Decomposition failed: {e}")

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

            response = await self.user_query_agent1.run(task=user_prompt)
            content = response.messages[-1].content
            return content
        except Exception as e:
            logger.exception("Failed to answer query: %s", str(e))
            raise


    async def process_user_query(self, user_query: str,top_k:int = 10) -> str:
        """
        Full retrieval pipeline:
        - Decompose query
        - Generate embedding
        - Semantic search
        - Rerank
        - Get LLM answer
        :param user_query: Original query from user
        :return: Final answer from LLM

        Args:
            top_k:
        """
        try:
            # Step 1: Decompose the user query
            logger.info("Decomposing user query...")
            decomposed_queries = await self.query_decomposition(user_query)

            all_reranked_results = []

            for query in decomposed_queries:
                # Step 2: Embed the sub-query
                logger.info(f"Generating embedding for subquery: {query}")
                query_embedding = self.embed_text(query)

                # Step 3 & 4: Semantic search + Top-K results
                logger.info("Performing semantic search...")
                top_k_matches = self.semantic_search(query_embedding, top_k= top_k)

                # Extract texts for reranking
                candidate_texts = [json_content for _, json_content in top_k_matches]

                # Step 5: Rerank with cross-encoder
                logger.info("Reranking results...")
                reranked = self.rerank_with_cross_encoder(query, candidate_texts)

                all_reranked_results.extend(reranked)
                # Format final reranked content
                formatted_reranked = self.format_reranked_results(all_reranked_results)

                # Step 6: Get LLM answer
                logger.info("Generating final answer...")
                answer = await self.answer_query(user_query, formatted_reranked)

                return answer
        except Exception as e:
            logger.exception("Error processing user query: %s", str(e))
            raise

