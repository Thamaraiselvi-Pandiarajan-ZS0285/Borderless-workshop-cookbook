from abc import ABC, abstractmethod
from typing import List, Tuple

from sqlalchemy.engine.base import Engine
from sqlalchemy.orm.session import sessionmaker


class RetrievalInterface(ABC):
    """
    Abstract interface defining required methods for all retrieval backends.
    """

    def __init__(self, db_engine: Engine, db_session: sessionmaker):
        self.db_engine = db_engine
        self.db_session = db_session

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector for a given text.

        Args:
            text (str): Input text to embed

        Returns:
            List[float]: Embedding vector
        """
        pass

    @abstractmethod
    def semantic_search(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[float, dict]]:
        """
        Perform semantic search using a query embedding.

        Args:
            query_embedding (List[float]): Embedding vector of the query
            top_k (int): Number of top results to return

        Returns:
            List[Tuple[float, dict]]: List of (score, result) pairs
        """
        pass


    @abstractmethod
    async def query_decomposition(self, user_query: str) -> str:
        """
        Decomposes a user query into simpler subqueries.

        Args:
            user_query (str): The input query from the user.

        Returns:
            str: A json string of decomposed subqueries.
        """
        pass

    @abstractmethod
    def rerank_with_cross_encoder(self, query: str, candidate_texts: List[str]) -> List[Tuple[float, str]]:
        """
        Rerank candidate texts using a cross-encoder model.

        Args:
            query (str): User query
            candidate_texts (List[str]): List of candidate text strings

        Returns:
            List[Tuple[float, str]]: Reranked list of (score, text)
        """
        pass

    @abstractmethod
    async def answer_query(self, query_input: str, formatted_reranked_results: str) -> str:
        """
        Generate an answer to a user query using a custom assistant agent.

        Args:
            query_input (str): User question
            formatted_reranked_results (str): Contextual information from retrieval

        Returns:
            str: Assistant's response
        """
        pass

    @abstractmethod
    async def process_user_query(self, user_query: str, top_k: int = 10) -> str:
        """
        Full retrieval pipeline:
        - Decompose query
        - Generate embedding
        - Semantic search
        - Rerank
        - Get LLM answer

        Args:
            user_query (str): Original query from user
            top_k (int): Number of results to fetch

        Returns:
            str: Final answer from LLM
        """
        pass

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1 (List[float]): First vector
            vec2 (List[float]): Second vector

        Returns:
            float: Cosine similarity score
        """
        try:
            import numpy as np
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        except Exception as e:
            raise RuntimeError(f"Cosine similarity computation failed: {e}")