import json
import os
import uuid
from http.client import HTTPException

from pdf2image import convert_from_path

from backend.app.core.email_to_pdf_converter import HTMLEmailToPDFConverter
from backend.app.core.file_operations import FileToBase64
from backend.app.core.paper_itemizer import PaperItemizer
from backend.app.request_handler.email_request import EmailClassificationRequest, EmailClassifyImageRequest
from backend.app.request_handler.metadata_extraction import EmailImageRequest
from backend.config.dev_config import DEFAULT_IMAGE_FORMAT
from backend.utils.file_utils import FilePathUtils
from main import app, do_classify_via_vlm, upload_email_images, ingest_embedding


@app.post("/api/all-in-one")
async def test(email_file: EmailClassificationRequest):
    try:

        email_data = email_file.model_dump()  #to do: if body is html, convert to pdf using pdf plumber

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

        email_image_request= []
        summaries = []
        classify_image_request_data = {"imagedata": [], "json_data": email_file}
        for result in results:
            input_data = result["filePath"]
            file_extension = result["fileExtension"]
            file_name = result["fileName"]
            classify_image_request_data["imagedata"].append({"input_path":input_data, "file_name":file_name, "file_extension":file_extension})
            classify_image_request = EmailClassifyImageRequest.model_validate(classify_image_request_data)
            classify_via_llm = await do_classify_via_vlm(classify_image_request)
            category = classify_via_llm.classification
            summary = classify_via_llm.summary
            summaries.append(summary)
            email_image_request.append({"input":input_data, "file_name":file_name, "file_extension":file_extension, "category": category})

        email_request = EmailImageRequest(data=email_image_request)

        response = await upload_email_images(email_request)

        for result in response["results"]:
            subject = result["extracted_metadata"]["subject"]
            full_email_text = result["extracted_metadata"]["full_email_text"]
            combined_text = f"Subject: {subject}\n\n{full_email_text}\nAttachment Summary:{summaries}"
            await ingest_embedding(combined_text,response)

        return response

    except ValueError as ve:
        raise HTTPException(status_code=400, detail={"error": str(ve)})

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail={"error": "Invalid response format from the LLM"})

    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "Internal server error", "details": str(e)})