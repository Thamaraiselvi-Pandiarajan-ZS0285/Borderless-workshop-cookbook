from backend.app.core.semantic_search import SemanticSearch
from backend.app.core.reranker import Reranker
from backend.app.core.rag_builder import RAGBuilder
from backend.db.db_helper.db_Initializer import DbInitializer
from backend.config.db_config import *

class Retriever:
    def __init__(self):
        db_init = DbInitializer(
            POSTGRESQL_DRIVER_NAME, POSTGRESQL_HOST, POSTGRESQL_DB_NAME,
            POSTGRESQL_USER_NAME, POSTGRESQL_PASSWORD, POSTGRESQL_PORT_NO
        )

        db_init.db_create_engin()
        
        self.search = SemanticSearch(db_init.db_create_session())
        self.reranker = Reranker()
        self.rag = RAGBuilder()

    def retrieve(self, query: str, top_k=5):
        initial_results = self.search.search(query, top_k=top_k)
        reranked = self.reranker.rerank(query, initial_results)
        return self.rag.build_context(reranked)
