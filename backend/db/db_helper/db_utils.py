from sqlalchemy import Engine, MetaData, inspect
from sqlalchemy.sql.ddl import CreateSchema
import logging

from backend.models.metadata_extraction_json_embedding import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Dbutils:
    def __init__(self, db_engin: Engine, schema_names:list):
        self.db_engin = db_engin
        self.schema_names = schema_names
    def create_all_schema(self):
        with self.db_engin.connect() as conn:
            for schema_name in self.schema_names:
                conn.execute(CreateSchema(schema_name, if_not_exists=True))
                conn.commit()
    def create_all_table(self):
        logger.info("Starting table creation...")
        try:
            Base.metadata.create_all(self.db_engin, checkfirst=True)
            inspector = inspect(self.db_engin)
            tables = inspector.get_table_names()
            print("Existing tables:", tables)
            logger.info("Tables created successfully.")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def print_all_tables(self):
        metadata = MetaData()
        metadata.reflect(bind=self.db_engin)

        tables = metadata.tables.keys()

        print("List of tables:")
        for table in tables:
            print(table)
