from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, cast
from sqlalchemy.sql.functions import func
from backend.app.core.embedding_store import EmbeddingGenerator
from backend.db.db_helper.db_Initializer import DbInitializer
from backend.config.db_config import *
from backend.models.all_db_models import EmailChunk
from pgvector.sqlalchemy import Vector

class SemanticSearch:
    def __init__(self, db_session:sessionmaker):
        self.embedder = EmbeddingGenerator()
        self.db_session = db_session

    def search(self, query: str, top_k=5):
        vec = self.embedder.embed([query])[0].embedding
        session = self.db_session()

        try:
            results = (
                session.query(
                    EmailChunk.content_id,
                    EmailChunk.chunk_text,
                    func.l2_distance(EmailChunk.embedding, cast(vec, Vector)).label("distance")
                )
                .filter(EmailChunk.embedding != None).order_by("distance").limit(top_k)).all()
        finally:
            session.close()

        return [{"content": row[0], "text": row[1], "score": row[2]} for row in results]
