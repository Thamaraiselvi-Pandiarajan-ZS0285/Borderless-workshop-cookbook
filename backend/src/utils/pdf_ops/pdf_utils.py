import pdfplumber
import logging
from pathlib import Path
from typing import List, Dict, Union

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """
    Extracts and returns all text from a PDF file.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"File not found: {pdf_path}")

    try:
        logger.debug(f"Opening PDF: {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            all_text = "\n".join(page.extract_text() or '' for page in pdf.pages)
        logger.debug("Text extraction complete.")
        return all_text.strip()
    except Exception as e:
        logger.error(f"Failed to extract text: {e}")
        raise


def extract_text_with_bbox(pdf_path: Union[str, Path]) -> List[Dict]:
    """
    Extracts text elements with their bounding box data (x0, top, x1, bottom).
    Returns a list of dictionaries for each word.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"File not found: {pdf_path}")

    all_words = []

    try:
        logger.debug(f"Opening PDF for bbox text extraction: {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                words = page.extract_words()
                logger.debug(f"Page {page_num}: extracted {len(words)} words")
                all_words.extend(words)
        logger.debug("Text with bounding boxes extraction complete.")
        return all_words
    except Exception as e:
        logger.error(f"Failed to extract text with bounding boxes: {e}")
        raise
