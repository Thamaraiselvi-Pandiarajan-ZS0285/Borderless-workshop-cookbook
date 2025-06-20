from backend.app.core.email_processor import EmailProcessor
from backend.config.db_config import *
from backend.db.db_helper.db_Initializer import DbInitializer

db_init = DbInitializer(
    POSTGRESQL_DRIVER_NAME, POSTGRESQL_HOST, POSTGRESQL_DB_NAME,
    POSTGRESQL_USER_NAME, POSTGRESQL_PASSWORD, POSTGRESQL_PORT_NO
)

db_init.db_create_engin()

processor = EmailProcessor(db_init.db_create_session())
processor.process_emails(limit=5)
