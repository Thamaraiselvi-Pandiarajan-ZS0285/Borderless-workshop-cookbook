from backend.app.core.semantic_search import SemanticSearch
from backend.app.core.reranker import Reranker
from backend.app.core.rag_builder import RAGBuilder
from sqlalchemy.orm import sessionmaker

class Retriever:
    def __init__(self, db_session:sessionmaker):
        self.search = SemanticSearch(db_session)
        self.reranker = Reranker()
        self.rag = RAGBuilder()

    def retrieve(self, query: str, top_k=5):
        initial_results = self.search.search(query, top_k=top_k)
        reranked = self.reranker.rerank(query, initial_results)
        return self.rag.build_context(reranked)
