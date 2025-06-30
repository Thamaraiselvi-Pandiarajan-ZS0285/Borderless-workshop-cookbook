import base64
import logging
import os
import uuid
import mimetypes
import json
import re

from contextlib import asynccontextmanager
from json import JSONDecodeError
from typing import Dict, Any
from fastapi.encoders import jsonable_encoder

from fastapi import FastAPI, UploadFile, File, HTTPException, Body, status
from fastapi.responses import JSONResponse, FileResponse


from backend.app.core.classifier_agent import EmailClassifierProcessor
from backend.app.core.email_processing_pipelines.email_pipeline import EmailProcessingPipeline
from backend.app.core.embedder import Embedder
from backend.app.core.file_operations import FileToBase64
from backend.app.core.metadata_validation import MetadataValidatorAgent
from backend.app.core.ocr_agent import EmailOCRAgent
from backend.app.core.paper_itemizer import PaperItemizer
from backend.app.request_handler.email_request import EmailClassificationRequest, EmailClassifyImageRequest
from backend.app.core.user_query_handler import UserQueryAgent

from backend.app.request_handler.metadata_extraction import EmailImageRequest
from backend.app.request_handler.paper_itemizer import PaperItemizerRequest
from backend.app.response_handler.email_classifier_response import build_email_classifier_response, email_classify_response_via_vlm
from backend.app.response_handler.file_operations_reponse import build_encode_file_response
from backend.app.response_handler.paper_itemizer import build_paper_itemizer_response
from backend.config.db_config import *
from backend.db.db_helper.db_Initializer import DbInitializer
from backend.db.db_helper.db_utils import Dbutils
from backend.models.all_db_models import Base
from backend.prompts.summarization_prompt import TASK_VARIANTS
from backend.utils.base_64_operations import Base64Utils
from backend.app.core.summarization_agent import SummarizationAgent
from backend.utils.extract_data_from_file import AttachmentExtractor, split_into_pages
from backend.utils.file_utils import FilePathUtils
from backend.app.core.email_to_pdf_converter import HTMLEmailToPDFConverter


import logging

from autogen_core import EVENT_LOGGER_NAME

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Starting application lifespan...")

    try:
        logger.info("Initializing database connection...")

        db_init = DbInitializer(
            POSTGRESQL_DRIVER_NAME,
            POSTGRESQL_HOST,
            POSTGRESQL_DB_NAME,
            POSTGRESQL_USER_NAME,
            POSTGRESQL_PASSWORD,
            POSTGRESQL_PORT_NO
        )

        application.state.db_engine = db_init.db_create_engin()
        application.state.db_session = db_init.db_create_session()

        logger.info("Database engine and session created successfully.")
        logger.info("Initializing database helper...")
        db_helper = Dbutils(application.state.db_engine, SCHEMA_NAMES)

        logger.info("Creating all schemas...")
        db_helper.create_all_schema()
        logger.info("Schemas created successfully.")

        logger.info("Creating all tables...")
        db_helper.create_all_table()
        db_helper.print_all_tables()
        Base.metadata.create_all(application.state.db_engine)  # Create tables

        logger.info("Tables created successfully.")


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

app = FastAPI(title="Borderless Access", swagger_ui_parameters={"syntaxHighlight": {"theme": "obsidian"}}, lifespan=lifespan)
logger.info("FastAPI application initialized.")



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




@app.post("/api/paper-itemizer")
async def do_paper_itemizer(request: PaperItemizerRequest):
    logger.info("Received paper itemizer request for file: %s", request.file_name)

    try:
        paper_itemizer_object = PaperItemizer(
            input=request.input,
            file_name=request.file_name,
            extension = request.file_extension
        )

        result = paper_itemizer_object.do_paper_itemizer()

        logger.info("Successfully processed file: %s", request.file_name)
        return build_paper_itemizer_response(result, 200, "Paper itemization successful.")

    except base64.binascii.Error as decode_err:
        logger.error("Base64 decoding failed for file %s: %s", request.file_name, str(decode_err), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 input. Decoding failed."
        )

    except HTTPException as http_ex:
        logger.warning("HTTPException raised for file %s: %s", request.file_name, str(http_ex.detail), exc_info=True)
        raise http_ex

    except Exception as e:
        logger.exception("Unexpected error while processing file %s: %s", request.file_name, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred while processing the file."
        )


@app.post("/api/convert/email-to-pdf")
async def convert_email_to_pdf(email_file: UploadFile = File(...)) -> FileResponse:
    """
    Convert a JSON-formatted email to a PDF document.

    Args:
        email_file (UploadFile): A JSON file containing email fields like subject, sender, received_at, etc.

    Returns:
        FileResponse: The generated PDF file.
    """
    try:
        contents = await email_file.read()
        email_data = json.loads(contents)

        if not isinstance(email_data, dict):
            raise ValueError("The uploaded JSON must be an object.")

        file_utils = FilePathUtils(file=email_file, temp_dir=None)
        output_dir = file_utils.file_dir()
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        file_name = file_utils.get_file_name()
        pdf_path = os.path.join(output_dir, f"{file_name}.pdf")
        pdf_converter = HTMLEmailToPDFConverter()
        pdf_converter.convert_to_pdf(email_data, pdf_path)

        logger.info(f"✅ PDF generated at: {pdf_path}")
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename="converted_email.pdf"
        )

    except JSONDecodeError:
        logger.exception("❌ Uploaded file is not valid JSON.")
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid JSON object.")

    except ValueError as ve:
        logger.exception("❌ Validation error in uploaded email JSON.")
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        logger.exception("❌ Unexpected error during PDF conversion.")
        raise HTTPException(status_code=500, detail="Error processing email: " + str(e))

@app.post("/api/classify_email")
def do_classify(email: EmailClassificationRequest):

    try:
        processor = EmailClassifierProcessor()
        summarizer = SummarizationAgent()
        extractor = AttachmentExtractor()

        full_body = email.body
        attachment_summary:str=""
        email_and_attachment_summary:str=""

        if email.hasAttachments:
            # Step 1: Extract raw attachment content
            attachment_content = extractor.extract_many(email.attachments)

            # Step 2: Split into page-wise chunks
            pages = split_into_pages(attachment_content)

            # Step 3: Summarize each page individually
            page_summaries = []
            for idx, page in enumerate(pages):
                summary = summarizer.summarize_text(page)
                page_summaries.append(f"Page {idx+1} Summary:\n{summary}")

            # Step 4: Generate final summary from all page summaries
            combined_summaries_text = "\n\n".join(page_summaries)
            attachment_summary = summarizer.summarize_text(combined_summaries_text)

        # Step 5: Append final summary to the body
        full_body += "Attachment Summary\n\n" + attachment_summary
        # Step 6: Process classification
        email_classification = processor.process_email(email.subject, full_body)

        for label, variant in TASK_VARIANTS.items():
            summary_response = summarizer.summarize_text(full_body, variant)
            email_and_attachment_summary += f"{label}:\n{summary_response}\n\n"

        return build_email_classifier_response(email,email_classification,email_and_attachment_summary)

    except JSONDecodeError:
        raise HTTPException(status_code=500, detail={"error": "Invalid response format from the LLM"})
    except ValueError as ve:
        raise HTTPException(status_code=400, detail={"error": str(ve)})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Internal server error", "details": str(e)})


@app.post("/api/classify_email_vlm")
def do_classify_via_vlm(request: EmailClassifyImageRequest):
    try:
        classifier = EmailClassifierProcessor()
        summarizer = SummarizationAgent()
        extractor = AttachmentExtractor()

        full_email_content = request.json_data.body
        attachment_summary: str = ""
        email_and_attachment_summary: str = ""

        if request.json_data.hasAttachments:
            # Step 1: Extract raw attachment content
            attachment_content = extractor.extract_many(request.json_data.attachments)

            # Step 2: Split into page-wise chunks
            pages = split_into_pages(attachment_content)

            # Step 3: Summarize each page individually
            page_summaries = []
            for idx, page in enumerate(pages):
                summary = summarizer.summarize_text(page)
                page_summaries.append(f"Page {idx + 1} Summary:\n{summary}")

            # Step 4: Generate final summary from all page summaries
            combined_summaries_text = "\n\n".join(page_summaries)
            attachment_summary = summarizer.summarize_text(combined_summaries_text)

        # Step 5: Append final summary to the body
        full_email_content += "Attachment Summary\n\n" + attachment_summary
        extracted_texts = []
        # Step 6: classify the email via vlm
        for item in request.imagedata:
            try:
                # Resolve image to base64 string
                if os.path.exists(item.input_path):
                    base64_converter = FileToBase64(item.input_path)
                    base64_image = base64_converter.do_base64_encoding()
                else:
                    if not item.input_path.startswith("data:image") and len(item.input_path) < 100:
                        raise ValueError("Invalid base64 input or unreadable image path.")
                    base64_image = item.input_path
                extracted_text = classifier.classify_via_vlm(base64_image)
                extracted_texts.append(extracted_text)
            except Exception as e:
                logger.error(f"Failed to extract metadata for {item.file_name}: {e}", exc_info=True)
        for label, variant in TASK_VARIANTS.items():
            summary_response = summarizer.summarize_text(full_email_content, variant)
            email_and_attachment_summary += f"{label}:\n{summary_response}\n\n"
        return email_classify_response_via_vlm(request,extracted_texts,email_and_attachment_summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Internal server error", "details": str(e)})


@app.post("/api/extraction/metadata_extractor")
def upload_email_images(request: EmailImageRequest) -> Dict[str, Any]:
    """
    Accepts a list of email image inputs (either file path or base64 string),
    extracts metadata using OCR and LLM, and returns the result per file.

    Args:
        request (EmailImageRequest): List of image items with `file_name`, `file_extension`, and `input` fields.

    Returns:
        dict: Extraction result or error per file in the `results` list.
    """
    results = []
    ocr_agent = EmailOCRAgent()
    validator_agent = MetadataValidatorAgent()

    for item in request.data:
        try:
            # Resolve image to base64 string
            if os.path.exists(item.input):
                base64_converter = FileToBase64(item.input)
                base64_image = base64_converter.do_base64_encoding()
            else:
                if not item.input.startswith("data:image") and len(item.input) < 100:
                    raise ValueError("Invalid base64 input or unreadable image path.")
                base64_image = item.input
            extracted_text = ocr_agent.extract_text_from_base64(base64_image, item.category)
            cleaned_json_string = re.sub(r"^```json\s*|\s*```$", "", extracted_text.strip())
            validation_result = validator_agent.validate_metadata(cleaned_json_string, item.category)
            try:
                parsed_metadata = json.loads(cleaned_json_string)
                results.append({
                    "file_name": item.file_name,
                    "file_extension": item.file_extension,
                    "extracted_metadata": parsed_metadata,
                    "validation_result": validation_result
                })
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON from extracted text: {e}")


        except Exception as e:
            logger.error(f"Failed to extract metadata for {item.file_name}: {e}", exc_info=True)
            results.append({
                "file_name": item.file_name,
                "file_extension": item.file_extension,
                "error": str(e)
            })

    return {"results": results}

#
# @app.post("/query")
# def query(question: str):
#     try:
#         embedder = Embedder(app.state.db_engine, app.state.db_session)
#         answer = embedder.respond(question)
#         return {"answer": answer}
#     except ValueError as ve:
#         raise HTTPException(status_code=400, detail={"error": str(ve)})
#
#     except json.JSONDecodeError:
#         raise HTTPException(status_code=500, detail={"error": "Invalid response format from the LLM"})
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail={"error": "Internal server error", "details": str(e)})
#

@app.post("/ingest")
def ingest_embedding(email_content:str, response_json:Dict[str,list]):

    embedder = Embedder(app.state.db_engine, app.state.db_session)
    minified = embedder.minify_json(response_json)
    if not minified:
        raise HTTPException(status_code=400, detail="Invalid or empty JSON for embedding.")
    json_embedding = embedder.embed_text(minified)
    embedder.ingest_email_metadata_json( "sender@yahoo.com",minified,json_embedding)

    content_embedding = embedder.embed_text(email_content)
    embedder.ingest_email_for_content("sender@yahoo.com",email_content, content_embedding)


@app.post("/api/all-in-one")
async def test(email_file: EmailClassificationRequest):
    try:
        email_data = jsonable_encoder(email_file)

        if not isinstance(email_data, dict):
            raise HTTPException(status_code=400, detail="Invalid input format. Expecting JSON object.")

        pipeline = EmailProcessingPipeline()
        response = await pipeline.run_pipeline(email_data)
        return JSONResponse(content=response)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON response from LLM.")

    except Exception as e:
        logging.exception("Unhandled error occurred.")
        raise HTTPException(status_code=500, detail={"error": "Internal server error", "details": str(e)})

@app.post("/api/query-input")
async def user_query(user_query: str, top_k: int=10):
    user = UserQueryAgent()
    result = user.query_decomposition(user_query)

    embedder = Embedder(app.state.db_engine, app.state.db_session)
    query_embedding_result = embedder.embed_text(result)
    semantic_result = embedder.semantic_search(query_embedding_result, top_k=top_k*3)

    candidate_texts = [text for _, text in semantic_result]
    reranked = embedder.rerank_with_cross_encoder(user_query, candidate_texts)

    formatted_context = embedder.format_reranked_results(reranked)

    final_response = embedder.answer_query(user_query, formatted_context)

    return final_response