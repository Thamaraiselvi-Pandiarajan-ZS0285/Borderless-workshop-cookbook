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

from fastapi import FastAPI, UploadFile, File, HTTPException, Body, status
from fastapi.responses import JSONResponse, FileResponse

from backend.src.config.db_config import POSTGRESQL_DRIVER_NAME, POSTGRESQL_HOST, POSTGRESQL_DB_NAME, \
    POSTGRESQL_USER_NAME, POSTGRESQL_PASSWORD, POSTGRESQL_PORT_NO, SCHEMA_NAMES
from backend.src.config.dev_config import BASE_PATH, DEFAULT_IMAGE_FORMAT
from backend.src.controller.request_handler.email_request import EmailClassificationRequest, EmailClassifyImageRequest
from backend.src.controller.request_handler.metadata_extraction import EmailImageRequest
from backend.src.controller.request_handler.paper_itemizer import PaperItemizerRequest
from backend.src.controller.response_handler.email_classifier_response import build_email_classifier_response, \
    email_classify_response_via_vlm
from backend.src.controller.response_handler.file_operations_reponse import build_encode_file_response
from backend.src.controller.response_handler.paper_itemizer import build_paper_itemizer_response
from backend.src.core.base_agents.ocr_agent import EmailOCRAgent
from backend.src.core.email_classifier.classifier_agent import EmailClassifierProcessor
from backend.src.core.email_classifier.summarization_agent import SummarizationAgent
from backend.src.core.embeding.embedder import Embedder
from backend.src.core.ingestion.email_to_pdf_converter import HTMLEmailToPDFConverter
from backend.src.core.ingestion.paper_itemizer import PaperItemizer
from backend.src.core.meta_extractor.metadata_validation import MetadataValidatorAgent
from backend.src.core.retrival.user_query_handler import UserQueryAgent
from backend.src.db.db_helper.db_Initializer import DbInitializer
from backend.src.db.db_helper.db_utils import Dbutils
from backend.src.db.models.metadata_extraction_json_embedding import Base
from backend.src.prompts.summarization_prompt import TASK_VARIANTS
from backend.src.utils.base_64_ops.base_64_utils import encode_file_to_base64, is_valid_base64, decode_base64
from backend.src.utils.extract_data_from_file import AttachmentExtractor, split_into_pages
from backend.src.utils.file_ops.file_utils import file_save

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
        Base.metadata.create_all(application.state.db_engine)
        application.state.base_path = BASE_PATH
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
            file_path = file_save(file, file.filename, app.state.base_path)
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

        try:
            encoded_data = encode_file_to_base64(file_path)
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
        base64_validator =is_valid_base64(base64_string)
        if not base64_validator :
            logger.warning("Invalid base64 input received.")
            raise HTTPException(status_code=400, detail="Provided string is not valid base64.")

        output_dir = app.state.base_path
        safe_file_name = f"{file_name}_{uuid.uuid4().hex}{extension}"
        output_path = os.path.join(output_dir, safe_file_name)

        decoded_data = decode_base64(base64_string)
        file_save(decoded_data,file_name,output_dir)

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

        output_dir = app.state.base_path
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        file_name = email_file.filename
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
async def do_classify(email: EmailClassificationRequest):

    try:
        processor = EmailClassifierProcessor()
        summarizer = SummarizationAgent()
        extractor = AttachmentExtractor()

        full_body = email.body
        attachment_summary:str=""
        email_and_attachment_summary:str=""

        if email.hasAttachments:
            attachment_content = extractor.extract_many(email.attachments)

            pages = split_into_pages(attachment_content)

            page_summaries = []
            for idx, page in enumerate(pages):
                summary = summarizer.summarize_text(page)
                page_summaries.append(f"Page {idx+1} Summary:\n{summary}")

            combined_summaries_text = "\n\n".join(page_summaries)
            attachment_summary = await summarizer.summarize_text(combined_summaries_text)

        full_body += "Attachment Summary\n\n" + attachment_summary
        email_classification = await processor.process_email(email.subject, full_body)

        for label, variant in TASK_VARIANTS.items():
            summary_response =await summarizer.summarize_text(full_body, variant)
            email_and_attachment_summary += f"{label}:\n{summary_response}\n\n"

        return build_email_classifier_response(email,email_classification,email_and_attachment_summary)

    except JSONDecodeError:
        raise HTTPException(status_code=500, detail={"error": "Invalid response format from the LLM"})
    except ValueError as ve:
        raise HTTPException(status_code=400, detail={"error": str(ve)})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Internal server error", "details": str(e)})


@app.post("/api/classify_email_vlm")
async def do_classify_via_vlm(request: EmailClassifyImageRequest):
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
                summary = await summarizer.summarize_text(page)
                page_summaries.append(f"Page {idx + 1} Summary:\n{summary}")

            # Step 4: Generate final summary from all page summaries
            combined_summaries_text = "\n\n".join(page_summaries)
            attachment_summary = await summarizer.summarize_text(combined_summaries_text)

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
                extracted_text = await classifier.classify_via_vlm(base64_image)
                extracted_texts.append(extracted_text)
            except Exception as e:
                logger.error(f"Failed to extract metadata for {item.file_name}: {e}", exc_info=True)
        for label, variant in TASK_VARIANTS.items():
            summary_response =await summarizer.summarize_text(full_email_content, variant)
            email_and_attachment_summary += f"{label}:\n{summary_response}\n\n"
        return email_classify_response_via_vlm(request,extracted_texts,email_and_attachment_summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Internal server error", "details": str(e)})


@app.post("/api/extraction/metadata_extractor")
async def upload_email_images(request: EmailImageRequest) -> Dict[str, Any]:
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
            extracted_text = await ocr_agent.extract_text_from_base64(base64_image, item.category)
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


@app.post("/ingest")
async def ingest_embedding(email_content:str, response_json:Dict[str,list]):

    embedder = Embedder(app.state.db_engine, app.state.db_session)
    minified =await embedder.minify_json(response_json)
    if not minified:
        raise HTTPException(status_code=400, detail="Invalid or empty JSON for embedding.")
    json_embedding = await embedder.embed_text(minified)
    embedder.ingest_email_metadata_json( "sender@yahoo.com",minified,json_embedding)

    content_embedding = embedder.embed_text(email_content)
    embedder.ingest_email_for_content("sender@yahoo.com",email_content, content_embedding)


@app.post("/api/all-in-one")
async def test(email_file: EmailClassificationRequest):
    try:

        email_data = email_file.model_dump()  # to do: if body is html, convert to pdf using pdf plumber

        if not isinstance(email_data, dict):
            raise ValueError("The uploaded JSON must be an object.")



        file_utils = FilePathUtils(file=None, temp_dir=None)
        output_dir = file_utils.file_dir()
        os.makedirs(output_dir, exist_ok=True)
        file_name = str(uuid.uuid4())
        pdf_path = os.path.join(output_dir, f"{file_name}.pdf")

        # email to pdf
        pdf_converter = HTMLEmailToPDFConverter()
        pdf_saved_path= pdf_converter.convert_to_pdf(email_data, pdf_path)

        # encode
        base64_encoder = FileToBase64(pdf_path)
        encoded_data = base64_encoder.do_base64_encoding_by_file_path()

        # paper-itemizer
        paper_itemizer_object = PaperItemizer(
            input=encoded_data,
            file_name=file_name,
            extension=DEFAULT_IMAGE_FORMAT,
            pdf_file_path=pdf_saved_path
        )

        results = paper_itemizer_object.do_paper_itemizer()

        email_image_request = []
        summaries = []
        classify_image_request_data = {"imagedata": [], "json_data": email_file}
        for result in results:
            input_data = result["filePath"]
            file_extension = result["fileExtension"]
            file_name = result["fileName"]
            classify_image_request_data["imagedata"].append(
                {"input_path": input_data, "file_name": file_name, "file_extension": file_extension})
            classify_image_request = EmailClassifyImageRequest.model_validate(classify_image_request_data)
            classify_via_llm = await do_classify_via_vlm(classify_image_request)
            category = classify_via_llm.classification
            summary = classify_via_llm.summary
            summaries.append(summary)
            email_image_request.append(
                {"input": input_data, "file_name": file_name, "file_extension": file_extension, "category": category})

        email_request = EmailImageRequest(data=email_image_request)

        response = await upload_email_images(email_request)

        for result in response["results"]:
            subject = result["extracted_metadata"]["subject"]
            full_email_text = result["extracted_metadata"]["full_email_text"]
            combined_text = f"Subject: {subject}\n\n{full_email_text}\nAttachment Summary:{summaries}"
            await ingest_embedding(combined_text, response)

        return response

    except ValueError as ve:
        raise HTTPException(status_code=400, detail={"error": str(ve)})

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail={"error": "Invalid response format from the LLM"})

    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Internal server error", "details": str(e)})

@app.post("/api/query-input")
async def user_query(user_query: str, top_k: int=10):
    user = UserQueryAgent()
    result =  user.query_decomposition(user_query)

    embedder = Embedder(app.state.db_engine, app.state.db_session)
    query_embedding_result = embedder.embed_text(result)
    semantic_result =  embedder.semantic_search(query_embedding_result, top_k=top_k*3)

    candidate_texts = [text for _, text in semantic_result]
    reranked = embedder.rerank_with_cross_encoder(user_query, candidate_texts)

    formatted_context = embedder.format_reranked_results(reranked)

    final_response = await embedder.answer_query(user_query, formatted_context)

    return final_response