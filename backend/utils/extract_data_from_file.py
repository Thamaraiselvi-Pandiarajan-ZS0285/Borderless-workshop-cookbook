import base64
import io
from typing import List, Optional

import pdfplumber
from docx import Document
from bs4 import BeautifulSoup
from email import message_from_bytes, policy

from backend.app.request_handler.email_request import Attachment


def split_into_pages(content: str, max_chars: int = 2000) -> List[str]:
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
    def extract_many(self, attachments: List[Attachment], split_pages: bool = False, max_chars: int = 2000) -> str | List[str]:
        full_text = ""

        for attachment in attachments or []:
            try:
                content = self.extract(attachment)
                full_text += f"\n\n[Attachment Extract: {attachment.name}]\n{content}"
            except Exception as e:
                full_text += f"\n\n[Failed to extract from {attachment.name}]: {str(e)}"

        return split_into_pages(full_text, max_chars) if split_pages else full_text

    def extract(self, attachment: Attachment) -> str:
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
            return f"[Unsupported file type: {attachment.name}]"

    def _extract_from_pdf(self, base64_str: str) -> str:
        pdf_bytes = base64.b64decode(base64_str)
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])

    def _extract_from_docx(self, base64_str: str) -> str:
        docx_bytes = base64.b64decode(base64_str)
        document = Document(io.BytesIO(docx_bytes))
        return "\n".join([para.text for para in document.paragraphs])

    def _extract_from_txt(self, base64_str: str) -> str:
        return base64.b64decode(base64_str).decode("utf-8", errors="ignore")

    def _extract_from_html(self, base64_str: str) -> str:
        html_content = base64.b64decode(base64_str).decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator="\n")

    def _extract_from_eml(self, base64_str: str) -> str:
        eml_bytes = base64.b64decode(base64_str)
        msg = message_from_bytes(eml_bytes, policy=policy.default)
        if msg.is_multipart():
            parts = [part.get_payload(decode=True).decode("utf-8", errors="ignore")
                     for part in msg.walk()
                     if part.get_content_type() == "text/plain"]
            return "\n".join(parts)
        else:
            return msg.get_payload(decode=True).decode("utf-8", errors="ignore")
