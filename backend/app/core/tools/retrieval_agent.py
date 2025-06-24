from sqlalchemy.engine.base import Engine
from sqlalchemy.orm.session import sessionmaker

from backend.app.core.embedder import Embedder
from backend.app.core.user_query_handler import UserQueryAgent
from typing import Annotated
from pydantic import BaseModel, Field

class RetrievalTool:
    def __init__(self, db_engine:Engine, db_session :sessionmaker):
        self.db_engine = db_engine
        self.session = db_session

    def user_query(self,user_query: str, top_k: int=10):
        user = UserQueryAgent()
        result = user.query_decomposition(user_query)

        embedder = Embedder(self.db_engine,self.session)
        query_embedding_result = embedder.embed_text(result)
        semantic_result = embedder.semantic_search(query_embedding_result, top_k=top_k*3)

        candidate_texts = [text for _, text in semantic_result]
        reranked = embedder.rerank_with_cross_encoder(user_query, candidate_texts)

        formatted_context = embedder.format_reranked_results(reranked)

        final_response = embedder.answer_query(user_query, formatted_context)

        return final_response

class RetrievalToolInput(BaseModel):
    user_query: Annotated[str, Field(..., description="User's natural language question")]
    top_k: Annotated[int, Field(10, description="Number of top similar chunks to retrieve")]


def retrieval_tool_fn(user_query: str, top_k: int = 10, db_engine=None, db_session=None) -> str:
    tool = RetrievalTool(db_engine, db_session)
    return tool.user_query(user_query, top_k)