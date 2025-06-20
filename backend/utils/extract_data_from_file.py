import base64
import io
import logging
from typing import List
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup
from email import message_from_bytes, policy

from backend.app.request_handler.email_request import Attachment

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def split_into_pages(content: str, max_chars: int = 2000) -> List[str]:
    """
    Splits a long string into pages of limited character length.

    Args:
        content (str): Input text to split.
        max_chars (int): Maximum characters per page.

    Returns:
        List[str]: A list of paginated text strings.
    """
    lines = content.splitlines()
    pages = []
    current_page = []
    current_length = 0

    for line in lines:
        line_length = len(line)
        if current_length + line_length > max_chars:
            pages.append("\n".join(current_page))
            current_page = [line]
            current_length = line_length
        else:
            current_page.append(line)
            current_length += line_length

    if current_page:
        pages.append("\n".join(current_page))

    return pages


class AttachmentExtractor:
    """
    Extracts text content from various types of email attachments, including PDFs, DOCX, TXT, HTML, and EML.
    """

    def extract_many(self, attachments: List[Attachment], split_pages: bool = False, max_chars: int = 2000) -> str | List[str]:
        """
        Extracts and optionally paginates text from multiple attachments.

        Args:
            attachments (List[Attachment]): List of attachments to extract.
            split_pages (bool): Whether to split the result into pages.
            max_chars (int): Max characters per page (if split_pages is True).

        Returns:
            str | List[str]: Full combined extracted text or list of paginated pages.
        """
        full_text = ""

        for attachment in attachments or []:
            try:
                logger.info(f"Extracting content from attachment: {attachment.name}")
                content = self.extract(attachment)
                full_text += f"\n\n[Attachment Extract: {attachment.name}]\n{content}"
            except Exception as e:
                logger.warning(f"Failed to extract from {attachment.name}: {e}")
                full_text += f"\n\n[Failed to extract from {attachment.name}]: {str(e)}"

        return split_into_pages(full_text, max_chars) if split_pages else full_text

    def extract(self, attachment: Attachment) -> str:
        """
        Extracts text from a single attachment based on file type.

        Args:
            attachment (Attachment): The attachment to extract from.

        Returns:
            str: Extracted text or message for unsupported types.
        """
        name = attachment.name.lower()

        if name.endswith(".pdf"):
            return self._extract_from_pdf(attachment.contentBytes)
        elif name.endswith(".docx"):
            return self._extract_from_docx(attachment.contentBytes)
        elif name.endswith(".txt"):
            return self._extract_from_txt(attachment.contentBytes)
        elif name.endswith(".html") or attachment.contentType == "text/html":
            return self._extract_from_html(attachment.contentBytes)
        elif name.endswith(".eml"):
            return self._extract_from_eml(attachment.contentBytes)
        else:
            logger.warning(f"Unsupported file type: {attachment.name}")
            return f"[Unsupported file type: {attachment.name}]"

    @staticmethod
    def _extract_from_pdf(base64_str: str) -> str:
        """
        Extracts text from a base64-encoded PDF.

        Args:
            base64_str (str): Base64-encoded content.

        Returns:
            str: Extracted text.
        """
        pdf_bytes = base64.b64decode(base64_str)
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])

    def _extract_from_docx(self, base64_str: str) -> str:
        """
        Extracts text from a base64-encoded DOCX document.

        Args:
            base64_str (str): Base64-encoded content.

        Returns:
            str: Extracted text.
        """
        docx_bytes = base64.b64decode(base64_str)
        document = Document(io.BytesIO(docx_bytes))
        return "\n".join([para.text for para in document.paragraphs])

    def _extract_from_txt(self, base64_str: str) -> str:
        """
        Extracts text from a base64-encoded TXT file.

        Args:
            base64_str (str): Base64-encoded content.

        Returns:
            str: Decoded plain text.
        """
        return base64.b64decode(base64_str).decode("utf-8", errors="ignore")

    def _extract_from_html(self, base64_str: str) -> str:
        """
        Extracts text from a base64-encoded HTML file.

        Args:
            base64_str (str): Base64-encoded HTML.

        Returns:
            str: Extracted plain text.
        """
        html_content = base64.b64decode(base64_str).decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator="\n")

    def _extract_from_eml(self, base64_str: str) -> str:
        """
        Extracts plain text content from a base64-encoded EML email file.

        Args:
            base64_str (str): Base64-encoded EML.

        Returns:
            str: Extracted text content.
        """
        eml_bytes = base64.b64decode(base64_str)
        msg = message_from_bytes(eml_bytes, policy=policy.default)
        if msg.is_multipart():
            parts = [part.get_payload(decode=True).decode("utf-8", errors="ignore")
                     for part in msg.walk()
                     if part.get_content_type() == "text/plain"]
            return "\n".join(parts)
        else:
            return msg.get_payload(decode=True).decode("utf-8", errors="ignore")
