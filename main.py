"""
FastAPI Application for Email Processing and Document Management

This application provides endpoints for:
- Email classification and processing
- Document conversion and itemization
- Base64 encoding/decoding operations
- Metadata extraction from images
- Semantic search and query processing

Author: Borderless Access Team
Version: 1.0.0
"""

import base64
import json
import logging
import mimetypes
import os
import re
import uuid
from contextlib import asynccontextmanager
from json import JSONDecodeError
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Body, status
from fastapi.responses import JSONResponse, FileResponse
from pydantic import ValidationError

# Import configuration modules
from backend.src.config.db_config import (
    POSTGRESQL_DRIVER_NAME, POSTGRESQL_HOST, POSTGRESQL_DB_NAME,
    POSTGRESQL_USER_NAME, POSTGRESQL_PASSWORD, POSTGRESQL_PORT_NO, SCHEMA_NAMES
)
from backend.src.config.dev_config import BASE_PATH, DEFAULT_IMAGE_FORMAT

# Import request/response handlers
from backend.src.controller.request_handler.email_request import (
    EmailClassificationRequest, EmailClassifyImageRequest
)
from backend.src.controller.request_handler.metadata_extraction import EmailImageRequest
from backend.src.controller.request_handler.paper_itemizer import PaperItemizerRequest
from backend.src.controller.response_handler.email_classifier_response import (
    build_email_classifier_response, email_classify_response_via_vlm
)
from backend.src.controller.response_handler.file_operations_reponse import build_encode_file_response
from backend.src.controller.response_handler.paper_itemizer import build_paper_itemizer_response

# Import core processing modules
from backend.src.core.base_agents.ocr_agent import EmailOCRAgent
from backend.src.core.email_classifier.classifier_agent import EmailClassifierProcessor
from backend.src.core.email_classifier.summarization_agent import SummarizationAgent
from backend.src.core.embeding.embedder import Embedder
from backend.src.core.ingestion.email_to_pdf_converter import HTMLEmailToPDFConverter
from backend.src.core.ingestion.paper_itemizer import PaperItemizer
from backend.src.core.meta_extractor.metadata_validation import MetadataValidatorAgent
from backend.src.core.retrival.retrieval import RetrievalInterface
from backend.src.core.retrival.user_query_handler import UserQueryAgent

# Import database modules
from backend.src.db.db_helper.db_Initializer import DbInitializer
from backend.src.db.db_helper.db_utils import Dbutils
from backend.src.db.models.metadata_extraction_json_embedding import Base

# Import utility modules
from backend.src.prompts.summarization_prompt import TASK_VARIANTS
from backend.src.utils.base_64_ops.base_64_utils import (
    encode_file_to_base64, is_valid_base64, decode_base64
)
from backend.src.utils.extract_data_from_file import AttachmentExtractor, split_into_pages
from backend.src.utils.file_ops.file_utils import file_save

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Custom exception for database-related errors."""
    pass


class FileProcessingError(Exception):
    """Custom exception for file processing errors."""
    pass


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Manage the application lifespan, including database initialization and cleanup.

    Args:
        application (FastAPI): The FastAPI application instance

    Yields:
        None: Control to the application

    Raises:
        DatabaseError: If database initialization fails
    """
    logger.info("Starting application lifespan...")

    try:
        await _initialize_database(application)
        logger.info("Application initialization completed successfully.")
        yield
    except Exception as e:
        logger.error("Critical error during application startup: %s", str(e), exc_info=True)
        raise DatabaseError(f"Failed to initialize application: {str(e)}")
    finally:
        await _cleanup_database(application)


async def _initialize_database(application: FastAPI) -> None:
    """
    Initialize database connections and schema.

    Args:
        application (FastAPI): The FastAPI application instance

    Raises:
        DatabaseError: If database initialization fails
    """
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
        application.state.base_path = BASE_PATH

        logger.info("Database engine and session created successfully.")

        # Initialize database schema
        db_helper = Dbutils(application.state.db_engine, SCHEMA_NAMES)
        db_helper.create_all_schema()
        db_helper.create_all_table()
        Base.metadata.create_all(application.state.db_engine)

        logger.info("Database schema and tables created successfully.")

    except Exception as e:
        logger.error("Database initialization failed: %s", str(e), exc_info=True)
        raise DatabaseError(f"Database initialization failed: {str(e)}")


async def _cleanup_database(application: FastAPI) -> None:
    """
    Clean up database connections.

    Args:
        application (FastAPI): The FastAPI application instance
    """
    if hasattr(application.state, "db_engine"):
        logger.info("Closing database connection...")
        application.state.db_engine.dispose()
        logger.info("Database connection closed successfully.")


# Initialize FastAPI application
app = FastAPI(
    title="Borderless Access API",
    description="API for email processing, document management, and semantic search",
    version="1.0.0",
    swagger_ui_parameters={"syntaxHighlight": {"theme": "obsidian"}},
    lifespan=lifespan
)

logger.info("FastAPI application initialized.")


@app.get("/", tags=["Health Check"])
def health_check():
    """
    Health check endpoint to verify API availability.

    Returns:
        dict: Welcome message and API status
    """
    return {
        "message": "Welcome to Borderless Access API!",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.post("/api/encode", tags=["File Operations"])
async def encode_file(file: UploadFile = File(...)):
    """
    Encode an uploaded file to Base64 format.

    Args:
        file (UploadFile): The file to be encoded

    Returns:
        JSONResponse: Base64 encoded file data with metadata

    Raises:
        HTTPException: If file upload or encoding fails
    """
    if not file or not file.filename:
        logger.warning("No file uploaded for encoding.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided."
        )

    try:
        # Save uploaded file
        content = await file.read()
        file_path = file_save(content, file.filename, app.state.base_path)
        logger.info("File saved successfully: %s", file_path)

        # Encode file to Base64
        encoded_data = encode_file_to_base64(file_path)
        logger.info("File encoded successfully: %s", file.filename)

        return build_encode_file_response(encoded_data, 200, "File encoded successfully")

    except Exception as e:
        logger.error("Error encoding file '%s': %s", file.filename, str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to encode file: {str(e)}"
        )


@app.post("/api/decode", response_class=FileResponse, tags=["File Operations"])
async def decode_file(
    base64_string: str = Body(..., embed=True),
    file_name: str = Body("decoded_file", embed=True),
    extension: str = Body(".jpg", embed=True)
):
    """
    Decode a Base64 string to a file and return it for download.

    Args:
        base64_string (str): Base64 encoded file data
        file_name (str): Name for the decoded file
        extension (str): File extension for the decoded file

    Returns:
        FileResponse: The decoded file for download

    Raises:
        HTTPException: If Base64 decoding fails or file creation fails
    """
    try:
        # Validate Base64 input
        if not is_valid_base64(base64_string):
            logger.warning("Invalid Base64 input received.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provided string is not valid Base64."
            )

        # Generate safe file path
        output_dir = app.state.base_path
        safe_file_name = f"{file_name}_{uuid.uuid4().hex}{extension}"
        output_path = os.path.join(output_dir, safe_file_name)

        # Decode and save file
        decoded_data = decode_base64(base64_string)
        with open(output_path, 'wb') as f:
            f.write(decoded_data)

        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(output_path)
        media_type = mime_type or "application/octet-stream"

        logger.info("File decoded and saved successfully: %s", output_path)
        return FileResponse(
            path=output_path,
            filename=f"{file_name}{extension}",
            media_type=media_type
        )

    except base64.binascii.Error as e:
        logger.error("Base64 decoding failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Base64 data. Cannot decode."
        )
    except Exception as e:
        logger.error("Error decoding file: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while decoding file."
        )


@app.post("/api/paper-itemizer", tags=["Document Processing"])
async def itemize_paper(request: PaperItemizerRequest):
    """
    Process a document and split it into individual items/pages.

    Args:
        request (PaperItemizerRequest): Request containing file data and metadata

    Returns:
        JSONResponse: Itemized document data

    Raises:
        HTTPException: If document processing fails
    """
    logger.info("Processing paper itemization for file: %s", request.file_name)

    try:
        # Validate request
        if not request.input:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No input data provided."
            )

        # Initialize paper itemizer
        paper_itemizer = PaperItemizer(
            input=request.input,
            file_name=request.file_name,
            extension=request.file_extension
        )

        # Process document
        result = paper_itemizer.do_paper_itemizer()

        logger.info("Paper itemization completed successfully for: %s", request.file_name)
        return build_paper_itemizer_response(result, 200, "Paper itemization successful.")

    except base64.binascii.Error as e:
        logger.error("Base64 decoding failed for file '%s': %s", request.file_name, str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Base64 input. Decoding failed."
        )
    except Exception as e:
        logger.error("Error processing file '%s': %s", request.file_name, str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred while processing the file."
        )


@app.post("/api/convert/email-to-pdf", response_class=FileResponse, tags=["Email Processing"])
async def convert_email_to_pdf(email_file: UploadFile = File(...)) -> FileResponse:
    """
    Convert a JSON-formatted email to a PDF document.

    Args:
        email_file (UploadFile): JSON file containing email data

    Returns:
        FileResponse: Generated PDF file

    Raises:
        HTTPException: If email conversion fails
    """
    try:
        # Read and parse email data
        contents = await email_file.read()

        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file provided."
            )

        email_data = json.loads(contents)

        if not isinstance(email_data, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded JSON must be an object."
            )

        # Generate PDF
        output_dir = app.state.base_path
        os.makedirs(output_dir, exist_ok=True)

        file_name = email_file.filename or "email"
        pdf_path = os.path.join(output_dir, f"{file_name}.pdf")

        pdf_converter = HTMLEmailToPDFConverter()
        pdf_converter.convert_to_pdf(email_data, pdf_path)

        logger.info("PDF generated successfully: %s", pdf_path)
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename="converted_email.pdf"
        )

    except JSONDecodeError as e:
        logger.error("Invalid JSON in uploaded file: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid JSON object."
        )
    except Exception as e:
        logger.error("Error converting email to PDF: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing email: {str(e)}"
        )


@app.post("/api/classify_email", tags=["Email Processing"])
async def classify_email(email: EmailClassificationRequest):
    """
    Classify an email and generate summaries for different tasks.

    Args:
        email (EmailClassificationRequest): Email data for classification

    Returns:
        dict: Email classification results and summaries

    Raises:
        HTTPException: If email classification fails
    """
    try:
        # Initialize processors
        processor = EmailClassifierProcessor()
        summarizer = SummarizationAgent()
        extractor = AttachmentExtractor()

        full_body = email.body
        attachment_summary = ""
        email_and_attachment_summary = ""

        # Process attachments if present
        if email.hasAttachments and email.attachments:
            try:
                attachment_content = extractor.extract_many(email.attachments)
                pages = split_into_pages(attachment_content)

                # Summarize each page
                page_summaries = []
                for idx, page in enumerate(pages):
                    summary = await summarizer.summarize_text(page)
                    page_summaries.append(f"Page {idx+1} Summary:\n{summary}")

                # Generate final attachment summary
                combined_summaries_text = "\n\n".join(page_summaries)
                attachment_summary = await summarizer.summarize_text(combined_summaries_text)

            except Exception as e:
                logger.warning("Error processing attachments: %s", str(e))
                attachment_summary = "Error processing attachments"

        # Combine email content with attachment summary
        full_body += "\n\nAttachment Summary:\n" + attachment_summary

        # Classify email
        email_classification = await processor.process_email(email.subject, full_body)

        # Generate task-specific summaries
        for label, variant in TASK_VARIANTS.items():
            try:
                summary_response = await summarizer.summarize_text(full_body, variant)
                email_and_attachment_summary += f"{label}:\n{summary_response}\n\n"
            except Exception as e:
                logger.warning("Error generating summary for %s: %s", label, str(e))
                email_and_attachment_summary += f"{label}:\nError generating summary\n\n"

        return build_email_classifier_response(email, email_classification, email_and_attachment_summary)

    except ValidationError as e:
        logger.error("Validation error in email classification: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error("Error classifying email: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/api/classify_email_vlm", tags=["Email Processing"])
async def classify_email_with_vision(request: EmailClassifyImageRequest):
    """
    Classify email using Vision Language Model (VLM) for image analysis.

    Args:
        request (EmailClassifyImageRequest): Request containing email data and images

    Returns:
        dict: Email classification results with vision-based analysis

    Raises:
        HTTPException: If VLM classification fails
    """
    try:
        # Initialize processors
        classifier = EmailClassifierProcessor()
        summarizer = SummarizationAgent()
        extractor = AttachmentExtractor()

        full_email_content = request.json_data.body
        attachment_summary = ""
        email_and_attachment_summary = ""

        # Process attachments
        if request.json_data.hasAttachments and request.json_data.attachments:
            try:
                attachment_content = extractor.extract_many(request.json_data.attachments)
                pages = split_into_pages(attachment_content)

                page_summaries = []
                for idx, page in enumerate(pages):
                    summary = await summarizer.summarize_text(page)
                    page_summaries.append(f"Page {idx + 1} Summary:\n{summary}")

                combined_summaries_text = "\n\n".join(page_summaries)
                attachment_summary = await summarizer.summarize_text(combined_summaries_text)

            except Exception as e:
                logger.warning("Error processing attachments: %s", str(e))
                attachment_summary = "Error processing attachments"

        full_email_content += "\n\nAttachment Summary:\n" + attachment_summary

        # Process images with VLM
        extracted_texts = []
        for item in request.imagedata:
            try:
                # Resolve image to base64
                if os.path.exists(item.input_path):
                    base64_image = encode_file_to_base64(item.input_path)
                else:
                    if not item.input_path.startswith("data:image") and len(item.input_path) < 100:
                        raise ValueError("Invalid base64 input or unreadable image path.")
                    base64_image = item.input_path

                # Extract text using VLM
                extracted_text = await classifier.classify_via_vlm(base64_image)
                extracted_texts.append(extracted_text)

            except Exception as e:
                logger.error("Failed to extract text from image %s: %s", item.file_name, str(e))
                extracted_texts.append(f"Error processing {item.file_name}: {str(e)}")

        # Generate task-specific summaries
        for label, variant in TASK_VARIANTS.items():
            try:
                summary_response = await summarizer.summarize_text(full_email_content, variant)
                email_and_attachment_summary += f"{label}:\n{summary_response}\n\n"
            except Exception as e:
                logger.warning("Error generating summary for %s: %s", label, str(e))
                email_and_attachment_summary += f"{label}:\nError generating summary\n\n"

        return email_classify_response_via_vlm(request, extracted_texts, email_and_attachment_summary)

    except Exception as e:
        logger.error("Error in VLM email classification: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/api/extraction/metadata_extractor", tags=["Metadata Extraction"])
async def extract_metadata_from_images(request: EmailImageRequest) -> Dict[str, Any]:
    """
    Extract metadata from email images using OCR and LLM processing.

    Args:
        request (EmailImageRequest): List of image items for metadata extraction

    Returns:
        dict: Extraction results for each image

    Raises:
        HTTPException: If metadata extraction fails
    """
    if not request.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image data provided."
        )

    results = []
    ocr_agent = EmailOCRAgent()
    validator_agent = MetadataValidatorAgent()

    for item in request.data:
        try:
            # Resolve image to base64
            if os.path.exists(item.input):
                base64_image = encode_file_to_base64(item.input)
            else:
                if not item.input.startswith("data:image") and len(item.input) < 100:
                    raise ValueError("Invalid base64 input or unreadable image path.")
                base64_image = item.input

            # Extract text using OCR
            extracted_text = await ocr_agent.extract_text_from_base64(base64_image, item.category)

            # Clean extracted JSON
            cleaned_json_string = re.sub(r"^```json\s*|\s*```$", "", extracted_text.strip())

            # Validate metadata
            validation_result = validator_agent.validate_metadata(cleaned_json_string, item.category)

            # Parse and store results
            try:
                parsed_metadata = json.loads(cleaned_json_string)
                results.append({
                    "file_name": item.file_name,
                    "file_extension": item.file_extension,
                    "extracted_metadata": parsed_metadata,
                    "validation_result": validation_result
                })
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON from extracted text: {str(e)}")

        except Exception as e:
            logger.error("Failed to extract metadata for %s: %s", item.file_name, str(e), exc_info=True)
            results.append({
                "file_name": item.file_name,
                "file_extension": item.file_extension,
                "error": str(e)
            })

    return {"results": results}


@app.post("/api/ingest", tags=["Data Ingestion"])
async def ingest_embeddings(email_content: str, response_json: Dict[str, Any]):
    """
    Ingest email content and metadata into the embedding database.

    Args:
        email_content (str): The email content to embed
        response_json (Dict[str, Any]): Metadata to embed

    Returns:
        dict: Ingestion status

    Raises:
        HTTPException: If embedding ingestion fails
    """
    try:
        if not email_content.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email content cannot be empty."
            )

        if not response_json:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Response JSON cannot be empty."
            )

        embedder = Embedder(app.state.db_engine, app.state.db_session)

        # Process JSON metadata
        minified = await embedder.minify_json(response_json)
        if not minified:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or empty JSON for embedding."
            )

        # Generate and store embeddings
        json_embedding = await embedder.embed_text(minified)
        embedder.ingest_email_metadata_json("sender@example.com", minified, json_embedding)

        content_embedding = await embedder.embed_text(email_content)
        embedder.ingest_email_for_content("sender@example.com", email_content, content_embedding)

        logger.info("Successfully ingested embeddings for email content and metadata.")
        return {"status": "success", "message": "Embeddings ingested successfully"}

    except Exception as e:
        logger.error("Error ingesting embeddings: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest embeddings: {str(e)}"
        )


@app.post("/api/all-in-one", tags=["Integrated Processing"])
async def process_email_comprehensive(email_file: EmailClassificationRequest):
    """
    Comprehensive email processing pipeline including conversion, itemization,
    classification, metadata extraction, and embedding ingestion.

    Args:
        email_file (EmailClassificationRequest): Email data for comprehensive processing

    Returns:
        dict: Complete processing results

    Raises:
        HTTPException: If any step in the pipeline fails
    """
    try:
        email_data = email_file.model_dump()

        if not isinstance(email_data, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email data format."
            )

        # Step 1: Convert email to PDF
        output_dir = app.state.base_path
        os.makedirs(output_dir, exist_ok=True)
        file_name = str(uuid.uuid4())
        pdf_path = os.path.join(output_dir, f"{file_name}.pdf")

        pdf_converter = HTMLEmailToPDFConverter()
        pdf_saved_path = pdf_converter.convert_to_pdf(email_data, pdf_path)
        logger.info("Email converted to PDF: %s", pdf_saved_path)

        # Step 2: Encode PDF to base64
        encoded_data = encode_file_to_base64(pdf_path)

        # Step 3: Paper itemization
        paper_itemizer = PaperItemizer(
            input=encoded_data,
            file_name=file_name,
            extension=DEFAULT_IMAGE_FORMAT,
            pdf_file_path=pdf_saved_path
        )
        results = paper_itemizer.do_paper_itemizer()
        logger.info("Paper itemization completed with %d items", len(results))

        # Step 4: Process each itemized result
        email_image_request = []
        summaries = []

        for result in results:
            try:
                input_data = result["filePath"]
                file_extension = result["fileExtension"]
                result_file_name = result["fileName"]

                # Create classification request
                classify_image_request_data = {
                    "imagedata": [{
                        "input_path": input_data,
                        "file_name": result_file_name,
                        "file_extension": file_extension
                    }],
                    "json_data": email_file
                }

                classify_image_request = EmailClassifyImageRequest.model_validate(
                    classify_image_request_data
                )

                # Classify via VLM
                classify_via_llm = await classify_email_with_vision(classify_image_request)
                category = classify_via_llm.classification
                summary = classify_via_llm.summary
                summaries.append(summary)

                # Prepare for metadata extraction
                email_image_request.append({
                    "input": input_data,
                    "file_name": result_file_name,
                    "file_extension": file_extension,
                    "category": category
                })

            except Exception as e:
                logger.error("Error processing result item: %s", str(e))
                continue

        # Step 5: Extract metadata
        if email_image_request:
            email_request = EmailImageRequest(data=email_image_request)
            response = await extract_metadata_from_images(email_request)

            # Step 6: Ingest embeddings
            for result in response["results"]:
                if "extracted_metadata" in result:
                    try:
                        subject = result["extracted_metadata"].get("subject", "")
                        full_email_text = result["extracted_metadata"].get("full_email_text", "")
                        combined_text = f"Subject: {subject}\n\n{full_email_text}\nAttachment Summary: {' '.join(summaries)}"

                        await ingest_embeddings(combined_text, response)

                    except Exception as e:
                        logger.error("Error ingesting embeddings for result: %s", str(e))
                        continue

            logger.info("Comprehensive email processing completed successfully")
            return response
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No valid results from paper itemization"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in comprehensive email processing: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comprehensive processing failed: {str(e)}"
        )


@app.post("/api/query-input", tags=["Search & Query"])
async def process_user_query(user_query: str = Body(...), top_k: int = Body(10)):
    """
    Process user query using semantic search and return relevant results.

    Args:
        user_query (str): The user's search query
        top_k (int): Number of top results to return (default: 10)

    Returns:
        str: AI-generated response based on semantic search results

    Raises:
        HTTPException: If query processing fails
    """
    try:
        if not user_query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty."
            )

        if top_k <= 0 or top_k > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="top_k must be between 1 and 100."
            )

        # Initialize query processing components
        user_agent = UserQueryAgent()
        embedder = Embedder(app.state.db_engine, app.state.db_session)

        # Step 1: Query decomposition
        decomposed_query = user_agent.query_decomposition(user_query)
        logger.info("Query decomposed successfully")

        # Step 2: Generate query embedding
        query_embedding = await embedder.embed_text(decomposed_query)
        logger.info("query_embedding: ",query_embedding)

        # Step 3: Semantic search
        semantic_results = embedder.semantic_search(query_embedding, top_k=top_k * 3)

        if not semantic_results:
            return {
                "query": user_query,
                "response": "No relevant results found for your query.",
                "results_count": 0
            }

        # Step 4: Re-rank results using cross-encoder
        candidate_texts = [text for _, text in semantic_results]
        reranked_results = embedder.rerank_with_cross_encoder(user_query, candidate_texts)

        # Step 5: Format results for context
        formatted_context = embedder.format_reranked_results(reranked_results)

        # Step 6: Generate final response
        final_response = await embedder.answer_query(user_query, formatted_context)

        logger.info("Query processed successfully with %d results", len(semantic_results))

        return {
            "query": user_query,
            "response": final_response,
            "results_count": len(semantic_results),
            "top_k_used": top_k
        }

    except Exception as e:
        logger.error("Error processing user query '%s': %s", user_query, str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )

@app.post("/query", summary="Process a user query through decomposition, retrieval, reranking, and LLM answering.")
async def process_query(user_query: str):
    """
    Processes a natural language query and returns a relevant answer using:
    - Query decomposition
    - Semantic search
    - Cross-encoder reranking
    - LLM-based answer generation

    Example input:
    {
      "user_query": "Show me emails about Q4 sales report"
    }
    """
    embedder = Embedder(db_engine=app.state.db_engine, db_session=app.state.db_session)
    user_query_agent = UserQueryAgent()
    retrieval_interface = RetrievalInterface(embedder=embedder, user_query_agent=user_query_agent)

    if not retrieval_interface:
        logger.error("Retrieval system not initialized.")
        raise HTTPException(status_code=500, detail="System not initialized.")

    if not user_query or not isinstance(user_query, str) or not user_query.strip():
        raise HTTPException(status_code=400, detail="Invalid query input.")

    try:
        logger.info(f"Processing query: {user_query}")
        answer = await retrieval_interface.process_user_query(user_query)
        return {"query": user_query, "answer": answer}
    except Exception as e:
        logger.exception("Error processing query: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")



# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler to catch and log unhandled exceptions.

    Args:
        request: The FastAPI request object
        exc: The exception that was raised

    Returns:
        JSONResponse: Error response with details
    """
    logger.error("Unhandled exception in %s %s: %s",
                request.method, request.url.path, str(exc), exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error occurred",
            "path": request.url.path,
            "method": request.method
        }
    )


# Custom exception handlers
@app.exception_handler(DatabaseError)
async def database_exception_handler(request, exc: DatabaseError):
    """Handle database-related exceptions."""
    logger.error("Database error in %s %s: %s",
                request.method, request.url.path, str(exc))

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": "Database service unavailable",
            "error_type": "DatabaseError"
        }
    )


@app.exception_handler(FileProcessingError)
async def file_processing_exception_handler(request, exc: FileProcessingError):
    """Handle file processing exceptions."""
    logger.error("File processing error in %s %s: %s",
                request.method, request.url.path, str(exc))

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "File processing failed",
            "error_type": "FileProcessingError"
        }
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    """Handle validation exceptions."""
    logger.error("Validation error in %s %s: %s",
                request.method, request.url.path, str(exc))

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": "Validation failed",
            "error_type": "ValidationError"
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )