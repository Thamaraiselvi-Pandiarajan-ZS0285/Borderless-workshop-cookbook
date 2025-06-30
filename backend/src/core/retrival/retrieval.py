import logging
from typing import List

from backend.src.core.embeding.embedder import Embedder
from backend.src.core.retrival.user_query_handler import UserQueryAgent

logger = logging.getLogger(__name__)

class RetrievalInterface:
    def __init__(self, embedder:Embedder, user_query_agent: UserQueryAgent):
        self.embedder = embedder
        self.user_query_agent = user_query_agent

    async def process_user_query(self, user_query: str) -> str:
        """
        Full retrieval pipeline:
        - Decompose query
        - Generate embedding
        - Semantic search
        - Rerank
        - Get LLM answer
        :param user_query: Original query from user
        :return: Final answer from LLM
        """
        try:
            # Step 1: Decompose the user query
            logger.info("Decomposing user query...")
            decomposed_queries = await self.user_query_agent.query_decomposition(user_query)

            all_reranked_results = []

            for query in decomposed_queries:
                # Step 2: Embed the sub-query
                logger.info(f"Generating embedding for subquery: {query}")
                query_embedding = self.embedder.embed_text(query)

                # Step 3 & 4: Semantic search + Top-K results
                logger.info("Performing semantic search...")
                top_k_matches = self.embedder.semantic_search(query_embedding, top_k=10)

                # Extract texts for reranking
                candidate_texts = [json_content for _, json_content in top_k_matches]

                # Step 5: Rerank with cross-encoder
                logger.info("Reranking results...")
                reranked = self.embedder.rerank_with_cross_encoder(query, candidate_texts)

                all_reranked_results.extend(reranked)
                # Format final reranked content
                formatted_reranked = self.embedder.format_reranked_results(all_reranked_results)

                # Step 6: Get LLM answer
                logger.info("Generating final answer...")
                answer = await self.embedder.answer_query(user_query, formatted_reranked)

                return answer
        except Exception as e:
            logger.exception("Error processing user query: %s", str(e))
            raise

