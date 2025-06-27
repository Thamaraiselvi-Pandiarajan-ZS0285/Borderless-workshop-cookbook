import os
import uuid
import logging

from fastapi import HTTPException

from backend.src.config.dev_config import BASE_PATH, IMAGE_RESOLUTION
from backend.src.utils.base_64_ops.base_64_utils import is_valid_base64, encode_file_to_base64, decode_base64
from backend.src.utils.file_ops.file_utils import file_save
from backend.src.utils.pdf_ops.paper_itemizer import paper_itemizer

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
        self.output_dir = BASE_PATH

    def do_base64_to_pdf_conversion(self) -> str:
        """Convert base64 input to PDF file and return file path."""
        logger.info("Starting base64 to PDF conversion.")
        if not is_valid_base64(self.input):
            logger.warning("Invalid base64 input provided.")
            raise HTTPException(status_code=400, detail="Provided string is not valid base64.")

        try:

            safe_file_name = f"{self.file_name}_{uuid.uuid4().hex}.pdf"
            output_path = os.path.join(self.output_dir, safe_file_name)

            decoded_data = decode_base64(self.input)
            file_path = file_save(decoded_data, self.file_name, self.output_dir)
            logger.info("Base64 converted to file successfully: %s", output_path)
            self.pdf_file_path = output_path
            return output_path

        except Exception as e:
            logger.exception("Failed to convert base64 to file.")
            raise HTTPException(status_code=500, detail="Failed to decode base64 input to PDF.")


    def do_encoder(self) -> list:
        """Encode image files to base64 and build the response list."""
        logger.info("Starting encoding of image files to base64.")
        try:
            file_response = []
            for file_path in self.file_paths:
                encoded_data = encode_file_to_base64(file_path)
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
        paper_itemizer(self.pdf_file_path,self.file_name,self.extension)
        return self.do_encoder()
