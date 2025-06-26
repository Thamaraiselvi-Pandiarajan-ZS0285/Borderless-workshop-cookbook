import json
import logging
import pathlib
import uuid
import re
from http.client import HTTPException
from typing import Dict, Any, Tuple
from autogen_agentchat.messages import TextMessage

from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from backend.app.core.classifier_agent import EmailClassifierProcessor
from backend.app.core.email_to_pdf_converter import HTMLEmailToPDFConverter
from backend.app.core.embedder import Embedder
from backend.app.core.file_operations import FileToBase64
from backend.app.core.metadata_consolidator import MetadataConsolidatorAgent
from backend.app.core.metadata_validation import MetadataValidatorAgent
from backend.app.core.ocr_agent import EmailOCRAgent
from backend.app.core.paper_itemizer import PaperItemizer
from backend.app.core.summarization_agent import SummarizationAgent
from backend.app.request_handler.email_request import EmailClassifyImageRequest
from backend.app.request_handler.metadata_extraction import EmailImageRequest
from backend.app.request_handler.paper_itemizer import PaperItemizerRequest
from backend.app.response_handler.email_classifier_response import email_classify_response_via_vlm
from backend.app.response_handler.paper_itemizer import build_paper_itemizer_response
from backend.config.dev_config import *
from backend.prompts.summarization_prompt import TASK_VARIANTS
from backend.utils.extract_data_from_file import AttachmentExtractor, split_into_pages
from backend.utils.file_utils import FilePathUtils
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker


import logging

from autogen_core import EVENT_LOGGER_NAME

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

class EmailProcessingPipeline:
    def __init__(self):
        self.model_client = AzureOpenAIChatCompletionClient(
            model=AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_ad_token=AZURE_OPENAI_API_KEY,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        )
        # Tool registration
        self.tools = [
            FunctionTool(
                func=self.convert_email_data_to_pdf,
                name="convert_email_data_to_pdf",
                description="Convert HTML email data to a PDF path"
            ),
            FunctionTool(
                func=self.do_encode_via_path,
                name="do_encode_via_path",
                description="Base64 encode a file by path"
            ),
            FunctionTool(
                func=self.do_paper_itemizer,
                name="do_paper_itemizer",
                description="Run paper itemizer on encoded file"
            ),
            FunctionTool(
                func=self.do_classify_via_vlm,
                name="do_classify_via_vlm",
                description="Classify email into types"
            ),
            FunctionTool(
                func=self.build_email_image_request,
                name="build_email_image_request",
                description="Build request for uploading email images"
            ),
            FunctionTool(
                func=self.upload_email_images,
                name="upload_email_images",
                description="Extract metadata from images"
            ),
            FunctionTool(
                func=self.ingest_all_embeddings,
                name="ingest_all_embeddings",
                description="Ingest combined email text and attachment summary"
            ),
        ]

        self.agent = AssistantAgent(
            name="support_agent",
            model_client=self.model_client,
            tools=self.tools,
            reflect_on_tool_use=True,
            system_message="""
        You are a backend orchestration agent that receives structured email data.

        Process it in this order:

        1. Call convert_email_data_to_pdf(email_data). It returns (pdf_path, file_name).

        2. Call do_encode_via_path(pdf_path=pdf_path, file_name=file_name). This returns PaperItemizerRequest.

        3. Call do_paper_itemizer(PaperItemizerRequest). It returns results (list of image data).

        4. Call do_classify_via_vlm(results) to classify the email. It returns classification_result.

        5. Call build_email_image_request(results, classification_result). It returns email_image_request.

        6. Call upload_email_images(email_request=email_image_request). It returns metadata_response.

        7. Call ingest_all_embeddings(response=metadata_response, summary=classification_result['summary']).

        → Return metadata_response to the user.

        ⚠️ Follow this exact order. Do not skip steps. Use TERMINATE after step 7.
        """
        )

    async def run_pipeline(self, email_data: dict) -> dict:
        if not isinstance(email_data, dict):
            raise ValueError(f"Input should be dict, got {type(email_data)}")

        try:
            text_message = TextMessage(
                content=json.dumps({"email_data": email_data}),
                source="User"
            )
            final_content = None
            async for event in self.agent.run_stream(task=text_message):
                if isinstance(event, TextMessage):
                    try:
                        # Attempt to parse the content as JSON
                        parsed_content = json.loads(event.content)
                        # If parsing is successful, process the content
                        final_content = parsed_content
                    except json.JSONDecodeError:
                        # If parsing fails, log the event and continue
                        logger.warning(f"Received non-JSON content: {event.content}")
                        continue
            return final_content
        except Exception as e:
            logging.exception("Pipeline execution failed.")
            raise e

    def do_paper_itemizer(self,request: PaperItemizerRequest):
        logger.info("Received paper itemizer request for file: %s", request.file_name)

        try:
            paper_itemizer_object = PaperItemizer(
                input=request.input,
                file_name=request.file_name,
                extension=request.file_extension
            )

            result = paper_itemizer_object.do_paper_itemizer()

            logger.info("Successfully processed file: %s", request.file_name)
            return build_paper_itemizer_response(result, 200, "Paper itemization successful.")

        except HTTPException as http_ex:
            logger.warning("HTTPException raised for file %s: %s", request.file_name, str(http_ex.detail),
                           exc_info=True)
            raise http_ex

    def do_classify_via_vlm(self, request: EmailClassifyImageRequest):
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
            return email_classify_response_via_vlm(request, extracted_texts, email_and_attachment_summary)

        except Exception as e:
            raise HTTPException(status_code=500, detail={"error": "Internal server error", "details": str(e)})

    def upload_email_images(self, request: EmailImageRequest) -> Dict[str, Any]:
        """
        Accepts a list of email image inputs (either file path or base64 string),
        extracts metadata using OCR and LLM, validates each, and then consolidates
        all JSONs into one metadata object if they are from the same email.

        Args:
            request (EmailImageRequest): List of image items with `file_name`, `file_extension`, and `input` fields.

        Returns:
            dict: Single consolidated metadata result or error.
        """
        ocr_agent = EmailOCRAgent()
        validator_agent = MetadataValidatorAgent()
        consolidator_agent = MetadataConsolidatorAgent()

        extracted_jsons = []
        errors = []
        if request.data[0].category.lower() in "rfp":
            email_category = "rfp"
        elif request.data[0].category.lower() in "bid-win":
            email_category = "bid-win"
        elif request.data[0].category.lower() in "rejection":
            email_category = "rejection"
        else:
            return {"error": "Human in the loop needed"}

        for item in request.data:
            try:
                # Convert file path or validate base64
                if os.path.exists(item.input):
                    base64_converter = FileToBase64(item.input)
                    base64_image = base64_converter.do_base64_encoding()
                else:
                    if not item.input.startswith("data:image") and len(item.input) < 100:
                        raise ValueError("Invalid base64 input or unreadable image path.")
                    base64_image = item.input

                # OCR extraction
                extracted_text = ocr_agent.extract_text_from_base64(base64_image, email_category)
                cleaned_json_string = re.sub(r"^```json\s*|\s*```$", "", extracted_text.strip())

                # Validate each extracted metadata
                validation_result = validator_agent.validate_metadata(cleaned_json_string, email_category)
                extracted_jsons.append(cleaned_json_string)

            except Exception as e:
                logger.error(f"Failed to extract metadata for {item.file_name}: {e}", exc_info=True)
                errors.append({
                    "file_name": item.file_name,
                    "file_extension": item.file_extension,
                    "error": str(e)
                })

        if errors:
            return {"errors": errors}

        # Consolidate all validated JSON strings into one final metadata
        try:
            consolidated_metadata = consolidator_agent.consolidate(extracted_jsons, category=email_category)
            return {
                "consolidated_metadata": consolidated_metadata
            }
        except Exception as e:
            logger.error(f"Metadata consolidation failed: {e}", exc_info=True)
            return {"error": f"Consolidation failed: {str(e)}"}

    def ingest_embedding(self, email_content: str, response_json: Dict[str, list], db_engine: Engine, db_session: sessionmaker):

        embedder = Embedder(db_engine, db_session)
        minified = embedder.minify_json(response_json)
        if not minified:
            raise HTTPException(status_code=400, detail="Invalid or empty JSON for embedding.")
        json_embedding = embedder.embed_text(minified)
        embedder.ingest_email_metadata_json("sender@yahoo.com", minified, json_embedding)

        content_embedding = embedder.embed_text(email_content)
        embedder.ingest_email_for_content("sender@yahoo.com", email_content, content_embedding)

    def build_email_image_request(self, results: list, classification_result: dict) -> dict:
        email_image_request = []
        for result in results:
            email_image_request.append({
                "input": result["filePath"],
                "file_name": result["fileName"],
                "file_extension": result["fileExtension"],
                "category": classification_result["classification"]
            })
        return {"data": email_image_request}

    def ingest_all_embeddings(self, response: dict, summary: str) -> dict:
        for result in response["results"]:
            metadata = result["extracted_metadata"]
            subject = metadata["subject"]
            full_email_text = metadata["full_email_text"]
            combined_text = f"Subject: {subject}\n\n{full_email_text}\nAttachment Summary:{summary}"
            self.ingest_embedding(combined_text, response)
        return {"status": "success"}

    def convert_email_data_to_pdf(self, email_data: dict) -> dict:
        if not isinstance(email_data, dict):
            raise ValueError("The input must be a dictionary.")

        file_utils = FilePathUtils(file=None, temp_dir=None)
        output_dir = file_utils.file_dir()
        os.makedirs(output_dir, exist_ok=True)
        file_name = str(uuid.uuid4())
        pdf_path = os.path.join(output_dir, f"{file_name}.pdf")

        pdf_converter = HTMLEmailToPDFConverter()
        pdf_converter.convert_to_pdf(email_data, pdf_path)

        logger.info(f"✅ PDF generated at: {pdf_path}")
        return {
            "pdf_path": pdf_path,
            "file_name": file_name
        }

    def do_encode_via_path(self, path:pathlib.Path|str, file_name:str) -> PaperItemizerRequest:

        # Double-check path exists
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="File not found")

        # Perform encoding
        try:
            base64_encoder = FileToBase64(str(path))
            encoded_data = base64_encoder.do_base64_encoding_by_file_path()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Encoding failed: {e}")
        paper_itemizer_object = PaperItemizerRequest(
            input=encoded_data,
            file_name=file_name,
            file_extension=DEFAULT_IMAGE_FORMAT)
        return paper_itemizer_object



