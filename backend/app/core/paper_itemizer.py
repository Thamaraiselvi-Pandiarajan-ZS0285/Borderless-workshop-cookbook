import os
import uuid
import logging

import pdfplumber
from fastapi import HTTPException

from backend.app.core.file_operations import FileToBase64
from backend.src.config import IMAGE_RESOLUTION, IMAGE_FORMATE
from backend.src.utils.base_64_operations import Base64Utils
from backend.src.utils.file_utils import FilePathUtils

# Logger configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PaperItemizer:
    def __init__(self, input: str, file_name: str, extension: str,pdf_file_path:str):
        self.input = input
        self.file_name = file_name
        self.extension = extension
        self.pdf_file_path = pdf_file_path
        self.file_paths = []

    def do_base64_to_pdf_conversion(self) -> str:
        """Convert base64 input to PDF file and return file path."""
        logger.info("Starting base64 to PDF conversion.")
        if not Base64Utils.is_valid_base64(self.input):
            logger.warning("Invalid base64 input provided.")
            raise HTTPException(status_code=400, detail="Provided string is not valid base64.")

        try:
            file_utils = FilePathUtils(file=None, temp_dir=None)
            output_dir = file_utils.file_dir()
            safe_file_name = f"{self.file_name}_{uuid.uuid4().hex}.pdf"
            output_path = os.path.join(output_dir, safe_file_name)

            FileToBase64.decode_base64_to_file(self.input, output_path)
            logger.info("Base64 converted to file successfully: %s", output_path)
            self.pdf_file_path = output_path
            return output_path

        except Exception as e:
            logger.exception("Failed to convert base64 to file.")
            raise HTTPException(status_code=500, detail="Failed to decode base64 input to PDF.")

    def paper_itemizer(self) -> list:
        """Convert PDF pages into high-resolution images."""
        logger.info("Starting PDF to image conversion: %s", self.pdf_file_path)
        if not self.pdf_file_path:
            raise HTTPException(status_code=400, detail="PDF file path not initialized.")

        try:
            with pdfplumber.open(self.pdf_file_path) as pdf:
                file_utils = FilePathUtils(file=None, temp_dir=None)
                output_dir = file_utils.file_dir()

                for i, page in enumerate(pdf.pages):
                    im = page.to_image(resolution=IMAGE_RESOLUTION)
                    safe_file_name = f"{self.file_name}_page_{i + 1}_{uuid.uuid4().hex}{self.extension}"
                    output_path = os.path.join(output_dir, safe_file_name)
                    rgb_image = im.original.convert("RGB")
                    rgb_image.save(output_path, format=IMAGE_FORMATE)

                    self.file_paths.append(output_path)
                    logger.info("Saved image for page %d: %s", i + 1, output_path)

                return self.file_paths

        except Exception as e:
            logger.exception("Failed during PDF to image conversion.")
            raise HTTPException(status_code=500, detail="Failed to convert PDF pages to images.")

    def do_encoder(self) -> list:
        """Encode image files to base64 and build the response list."""
        logger.info("Starting encoding of image files to base64.")
        try:
            file_response = []
            for file_path in self.file_paths:
                encoder = FileToBase64(file_path)
                encoded_data = encoder.do_base64_encoding_by_file_path()
                file_response.append({
                    "filePath": file_path,
                    "fileName": os.path.basename(file_path),
                    "fileExtension": self.extension,
                    "encode": encoded_data
                })
                logger.debug("Encoded file: %s", file_path)

            return file_response

        except Exception as e:
            logger.exception("Error encoding files to base64.")
            raise HTTPException(status_code=500, detail="Failed to encode image files to Base64.")

    def do_paper_itemizer(self) -> list:
        """End-to-end pipeline: base64 PDF -> images -> base64 images."""
        logger.info("Executing complete paper itemization process.")
        self.paper_itemizer()
        return self.do_encoder()
