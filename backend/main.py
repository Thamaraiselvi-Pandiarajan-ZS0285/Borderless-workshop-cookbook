import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.config.db_config import *;
from backend.db.db_helper.db_Initializer import DbInitializer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Borderless Access", swagger_ui_parameters={"syntaxHighlight": {"theme": "obsidian"}})
logger.info("FastAPI application initialized.")




@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Starting application lifespan...")

    try:
        logger.info("Initializing database connection...")
        db_init = DbInitializer(
            POSTGRESQL_DRIVER_NAME, POSTGRESQL_HOST, POSTGRESQL_DB_NAME,
            POSTGRESQL_USER_NAME, POSTGRESQL_PASSWORD, POSTGRESQL_PORT_NO
        )

        application.state.db_engine = db_init.db_create_engin()
        application.state.db_session = db_init.db_create_session()
        logger.info("Database engine and session created successfully.")

    except Exception as e:
        logger.error("Error during database initialization: %s", str(e), exc_info=True)
        raise

    try:
        yield
    finally:
        if hasattr(application.state, "db_engine"):
            logger.info("Closing database connection...")
            application.state.db_engine.dispose()
            logger.info("Database connection closed.")


@app.get("/")
def home():
    return {"message": "Welcome to Borderless Access!"}