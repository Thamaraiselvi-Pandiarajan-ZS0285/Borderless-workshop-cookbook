import base64
import logging
import os
import uuid
import mimetypes

from contextlib import asynccontextmanager
from pyexpat.errors import messages

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse
from backend.app.core.file_operations import FileToBase64
from backend.app.response_handler.file_operations_reponse import build_encode_file_response
from backend.config.db_config import *
from backend.db.db_helper.db_Initializer import DbInitializer
from backend.utils.base_64_operations import Base64Utils
from backend.utils.file_utils import FilePathUtils

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


@app.post("/api/encode")
async def do_encode(file: UploadFile = File(...)):
    try:
        if not file or not file.filename:
            logger.warning("No file uploaded.")
            raise HTTPException(status_code=400, detail="No file provided.")

        try:
            file_utils = FilePathUtils(file)
            file_path = file_utils.file_path_based_file_save()
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

        try:
            base64_encoder = FileToBase64(file_path)
            encoded_data = base64_encoder.do_base64_encoding_by_file_path()
        except Exception as e:
            logger.error(f"Error encoding file: {e}")
            raise HTTPException(status_code=500, detail="Failed to encode file to Base64.")

        return build_encode_file_response(encoded_data, 200, "File encoded successfully")

    except HTTPException as http_err:
        return JSONResponse(
            status_code=http_err.status_code,
            content={"statusCode": http_err.status_code, "message": http_err.detail, "response": None}
        )

    except Exception as unhandled:
        logger.error(f"Error encoding file: {unhandled}")
        return JSONResponse(
            status_code=500,
            content={"statusCode": 500, "message": "unhandled exception occurred", "response": unhandled}
        )


@app.post("/api/decode", response_class=FileResponse)
async def decode_file(
    base64_string: str = Body(..., embed=True),
    file_name: str = Body("decoded_file", embed=True),
    extension: str = Body(".jpg", embed=True)
):
    try:
        base64_validator = Base64Utils.is_valid_base64(base64_string)
        if not base64_validator :
            logger.warning("Invalid base64 input received.")
            raise HTTPException(status_code=400, detail="Provided string is not valid base64.")

        file_utils = FilePathUtils(file=None, temp_dir=None)
        output_dir = file_utils.file_dir()
        safe_file_name = f"{file_name}_{uuid.uuid4().hex}{extension}"
        output_path = os.path.join(output_dir, safe_file_name)

        FileToBase64.decode_base64_to_file(base64_string, output_path)

        mime_type, _ = mimetypes.guess_type(output_path)
        media_type = mime_type or "application/octet-stream"

        logger.info(f"Decoded file saved at {output_path}, returning as download.")
        return FileResponse(
            path=output_path,
            filename=f"{file_name}{extension}",
            media_type=media_type
        )

    except HTTPException as http_ex:
        raise http_ex

    except base64.binascii.Error as decode_err:
        logger.error("Base64 decoding failed: %s", str(decode_err), exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid base64 data. Cannot decode.")

    except Exception as e:
        logger.exception(f"Unhandled exception during decoding. {e}")
        raise HTTPException(status_code=500, detail="Internal server error while decoding file.")


