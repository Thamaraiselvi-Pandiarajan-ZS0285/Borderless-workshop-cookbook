import json
import os
import uuid
from json import JSONDecodeError
from typing import Dict, Any, re
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException
from sqlalchemy.engine.base import Engine

from backend.app.core.classifier_agent import EmailClassifierProcessor
from backend.app.core.email_to_pdf_converter import HTMLEmailToPDFConverter
from backend.app.core.embedder import Embedder
from backend.app.core.file_operations import FileToBase64
from backend.app.core.metadata_validation import MetadataValidatorAgent
from backend.app.core.ocr_agent import EmailOCRAgent
from backend.app.core.paper_itemizer import PaperItemizer
from backend.app.core.summarization_agent import SummarizationAgent
from backend.app.request_handler.email_request import EmailClassifyImageRequest
from backend.app.request_handler.metadata_extraction import EmailImageRequest
from backend.app.response_handler.email_classifier_response import build_email_classifier_response, \
    email_classify_response_via_vlm
from backend.config.dev_config import *
from backend.prompts.summarization_prompt import TASK_VARIANTS
from backend.utils.extract_data_from_file import AttachmentExtractor, split_into_pages
from backend.utils.file_utils import FilePathUtils
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)




class  EmailClassificationTool:
    def __init__(self):
        pass

    def __call__(self,email_file: EmailClassifyImageRequest):
        self.email_input = email_file

    def do_classify_via_vlm(self,request: EmailClassifyImageRequest):
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

    def upload_email_images(self,request: EmailImageRequest) -> Dict[str, Any]:
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

    def ingest_embedding(self,email_content: str, response_json: Dict[str, list], db_engine :Engine, db_session:sessionmaker):

        embedder = Embedder(db_engine, db_session)
        minified = embedder.minify_json(response_json)
        if not minified:
            raise HTTPException(status_code=400, detail="Invalid or empty JSON for embedding.")
        json_embedding = embedder.embed_text(minified)
        embedder.ingest_email_metadata_json("sender@yahoo.com", minified, json_embedding)

        content_embedding = embedder.embed_text(email_content)
        embedder.ingest_email_for_content("sender@yahoo.com", email_content, content_embedding)

    def test(self, email_input: EmailClassifyImageRequest):
        try:

            email_data = self.email_input.model_dump()  #to do: if body is html, convert to pdf using pdf plumber

            if not isinstance(email_data, dict):
                raise ValueError("The uploaded JSON must be an object.")

            file_utils = FilePathUtils(file=None, temp_dir=None)
            output_dir = file_utils.file_dir()
            os.makedirs(output_dir, exist_ok=True)
            file_name = str(uuid.uuid4())
            pdf_path = os.path.join(output_dir, f"{file_name}.pdf")
            #email to pdf
            pdf_converter = HTMLEmailToPDFConverter()
            pdf_converter.convert_to_pdf(email_data, pdf_path)
            #encode
            base64_encoder = FileToBase64(pdf_path)
            encoded_data = base64_encoder.do_base64_encoding_by_file_path()
            #paper-itemizer
            paper_itemizer_object = PaperItemizer(
                input=encoded_data,
                file_name=file_name,
                extension=DEFAULT_IMAGE_FORMAT
            )

            results = paper_itemizer_object.do_paper_itemizer()
            #classification
            # classification_result = self.do_classify()
            email_image_request = []
            summaries = []
            classify_image_request_data = {"imagedata": [], "json_data": email_file}

            for result in results:
                input_data = result["encode"]
                file_extension = result["fileExtension"]
                file_name = result["fileName"]
                classify_image_request_data["imagedata"].append(
                    {"input_path": input_data, "file_name": file_name, "file_extension": file_extension})
                classify_image_request = EmailClassifyImageRequest.model_validate(classify_image_request_data)
                classify_via_llm = self.do_classify_via_vlm(classify_image_request)
                category = classify_via_llm.classification
                summary = classify_via_llm.summary
                summaries.append(summary)
                email_image_request.append({"input":input_data, "file_name":file_name, "file_extension":file_extension, "category": category})

            email_request = EmailImageRequest(data=email_image_request)

            response = self.upload_email_images(email_request)
            # response["summary"] = classification_result.summary

            for result in response["results"]:
                subject = result["extracted_metadata"]["subject"]
                full_email_text = result["extracted_metadata"]["full_email_text"]
                combined_text = f"Subject: {subject}\n\n{full_email_text}\nAttachment Summary:{classification_result.summary}"
                self.ingest_embedding(combined_text,response)

            return response

        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail={"error": "Invalid response format from the LLM"})
        except ValueError as ve:
            raise HTTPException(status_code=400, detail={"error": str(ve)})
        except Exception as e:
            raise e
