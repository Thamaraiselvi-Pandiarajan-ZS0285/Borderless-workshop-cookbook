import os
import uuid
import logging

import pdfplumber
from fastapi import HTTPException  # Ensure this is imported if using FastAPI

from backend.src.config.dev_config import IMAGE_RESOLUTION, IMAGE_FORMATE, BASE_PATH

# Logger configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def paper_itemizer(pdf_file_path: str, file_name: str, extension: str = ".jpg") -> list:
    """
    Converts each page of the given PDF into high-resolution images and saves them.

    Args:
        pdf_file_path (str): Full path to the PDF file.
        file_name (str): Base name for the output image files.
        extension (str): File extension/format for output images (default is '.jpg').

    Returns:
        list: List of file paths to the saved images.

    Raises:
        HTTPException: If the PDF file path is invalid or the conversion fails.
    """
    logger.info("Starting PDF to image conversion: %s", pdf_file_path)

    if not pdf_file_path or not os.path.isfile(pdf_file_path):
        logger.error("Invalid or missing PDF file path: %s", pdf_file_path)
        raise HTTPException(status_code=400, detail="PDF file path is invalid or file does not exist.")

    file_paths = []

    try:
        with pdfplumber.open(pdf_file_path) as pdf:
            output_dir = BASE_PATH
            os.makedirs(output_dir, exist_ok=True)

            for i, page in enumerate(pdf.pages):
                try:
                    im = page.to_image(resolution=IMAGE_RESOLUTION)
                    safe_file_name = f"{file_name}_page_{i + 1}_{uuid.uuid4().hex}{extension}"
                    output_path = os.path.join(output_dir, safe_file_name)

                    rgb_image = im.original.convert("RGB")
                    rgb_image.save(output_path, format=IMAGE_FORMATE)

                    file_paths.append(output_path)
                    logger.info("Saved image for page %d: %s", i + 1, output_path)

                except Exception as page_ex:
                    logger.exception("Error processing page %d: %s", i + 1, str(page_ex))
                    continue

            return file_paths

    except Exception as e:
        logger.exception("Failed during PDF to image conversion.")
        raise HTTPException(status_code=500, detail="Failed to convert PDF pages to images.")
