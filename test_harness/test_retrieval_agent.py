from langchain_core.language_models import BaseLanguageModel

from backend.app.core.retrieval_agent import RetrievalAgent
from backend.db.db_helper.db_Initializer import DbInitializer
from backend.config.db_config import *

db_init = DbInitializer(
    POSTGRESQL_DRIVER_NAME, POSTGRESQL_HOST, POSTGRESQL_DB_NAME,
    POSTGRESQL_USER_NAME, POSTGRESQL_PASSWORD, POSTGRESQL_PORT_NO
)

db_engine = db_init.db_create_engin()
db_session = db_init.db_create_session()

agent = RetrievalAgent(db_session)
query = "Get all healthcare RFPs for India and summarize last month’s win rate?"
answer = agent.answer(query)
print(answer)
