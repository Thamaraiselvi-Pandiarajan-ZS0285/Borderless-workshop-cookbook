import os
import uuid
from http.client import HTTPException
import re
from typing import Dict, Any

from backend.app.core.email_to_pdf_converter import HTMLEmailToPDFConverter
from backend.app.core.paper_itemizer import PaperItemizer
from backend.app.response_handler.paper_itemizer import build_paper_itemizer_response
from backend.app.core.classifier_agent import EmailClassifierProcessor
from backend.app.core.summarization_agent import SummarizationAgent
from backend.app.request_handler.email_request import EmailClassifyImageRequest
from backend.app.response_handler.email_classifier_response import email_classify_response_via_vlm
from backend.app.request_handler.metadata_extraction import EmailImageRequest
from backend.app.core.ocr_agent import EmailOCRAgent
from backend.app.core.metadata_validation import MetadataValidatorAgent
from backend.app.core.metadata_consolidator import MetadataConsolidatorAgent
from backend.app.core.embedder import Embedder
from backend.app.core.file_operations import FileToBase64
from backend.utils.extract_data_from_file import AttachmentExtractor, split_into_pages
from backend.app.request_handler.paper_itemizer import PaperItemizerRequest
from backend.config.dev_config import DEFAULT_IMAGE_FORMAT
from backend.prompts.summarization_prompt import TASK_VARIANTS
from backend.utils.file_utils import FilePathUtils

# 🔧 Standalone Tool Functions

def convert_email_data_to_pdf(data: Dict[str, Any]) -> Dict[str, Any]:
    email_data = data.get("email_data", {})
    file_utils = FilePathUtils(file=None, temp_dir=None)
    output_dir = file_utils.file_dir()
    os.makedirs(output_dir, exist_ok=True)
    file_name = str(uuid.uuid4())
    pdf_path = os.path.join(output_dir, f"{file_name}.pdf")
    HTMLEmailToPDFConverter().convert_to_pdf(email_data, pdf_path)
    return {"pdf_path": pdf_path, "file_name": file_name}

def do_encode_via_path(data: Dict[str, Any]) -> Dict[str, Any]:
    path = data["path"]
    file_name = data["file_name"]
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found")
    encoded_data = FileToBase64(path).do_base64_encoding_by_file_path()
    return {"input": encoded_data, "file_name": file_name, "file_extension": DEFAULT_IMAGE_FORMAT}

def do_paper_itemizer(data: Dict[str, Any]) -> Dict[str, Any]:
    req = PaperItemizerRequest(**data)
    result = PaperItemizer(input=req.input, file_name=req.file_name, extension=req.file_extension).do_paper_itemizer()
    return build_paper_itemizer_response(result, 200, "Paper itemization successful.")

def do_classify_via_vlm(data: Dict[str, Any]) -> Dict[str, Any]:
    req = EmailClassifyImageRequest(**data)
    classifier = EmailClassifierProcessor()
    summarizer = SummarizationAgent()
    extractor = AttachmentExtractor()
    full = req.json_data.body
    attach_summary = ""
    if req.json_data.hasAttachments:
        pages = split_into_pages(extractor.extract_many(req.json_data.attachments))
        page_summaries = [f"Page {i+1} Summary:\n{summarizer.summarize_text(p)}" for i,p in enumerate(pages)]
        attach_summary = summarizer.summarize_text("\n\n".join(page_summaries))
        full += "\nAttachment Summary\n\n" + attach_summary
    extracted = []
    for item in req.imagedata:
        base64_img = FileToBase64(item.input_path).do_base64_encoding_by_file_path() \
            if os.path.exists(item.input_path) else item.input_path
        extracted.append(classifier.classify_via_vlm(base64_img))
    summary = "".join(f"{label}:\n{summarizer.summarize_text(full, variant)}\n\n" for label,variant in TASK_VARIANTS.items())
    return email_classify_response_via_vlm(req, extracted, summary)

def build_email_image_request(data: Dict[str, Any]) -> Dict[str, Any]:
    resp = data["paper_itemizer_response"]
    classification = data["classification_result"]
    return {"data": [{"input": r["filePath"], "file_name": r["fileName"], "file_extension": r["fileExtension"], "category": classification["classification"]} for r in resp["results"]]}

def upload_email_images(data: Dict[str, Any]) -> Dict[str, Any]:
    req = EmailImageRequest(**data)
    ocr = EmailOCRAgent()
    val = MetadataValidatorAgent()
    cons = MetadataConsolidatorAgent()
    extracted = []; errors = []
    for item in req.data:
        try:
            base64_img = FileToBase64(item.input).do_base64_encoding_by_file_path()
            cleaned = re.sub(r"^```json\s*|\s*```$", "", ocr.extract_text_from_base64(base64_img, item.category).strip())
            val.validate_metadata(cleaned, item.category)
            extracted.append(cleaned)
        except Exception as e:
            errors.append({"file_name": item.file_name, "error": str(e)})
    if errors:
        return {"errors": errors}
    return {"consolidated_metadata": cons.consolidate(extracted, category=req.data[0].category)}

def ingest_all_embeddings(data: Dict[str, Any]) -> Dict[str, Any]:
    resp = data["upload_response"]
    summary = data["classification_summary"]
    for r in resp.get("results", []):
        text = f"Subject: {r['extracted_metadata']['subject']}\n\n{r['extracted_metadata']['full_email_text']}\nAttachment Summary:{summary}"
        Embedder(None, None).ingest_email_for_content(text, text)
    return {"status": "success"}
