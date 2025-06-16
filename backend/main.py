import logging

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Borderless Access", swagger_ui_parameters={"syntaxHighlight": {"theme": "obsidian"}})
logger.info("FastAPI application initialized.")



@app.get("/")
def home():
    return {"message": "Welcome to Borderless Access!"}