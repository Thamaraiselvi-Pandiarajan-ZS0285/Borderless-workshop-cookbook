import os
import logging
from typing import Dict, Any
from abc import ABC, abstractmethod
from jinja2 import Environment, FileSystemLoader, Template
import pdfkit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EmailConverter(ABC):
    @abstractmethod
    def convert_to_pdf(self, email_data: Dict[str, Any], output_path: str) -> None:
        """Abstract method to convert email data to PDF."""
        pass


class HTMLEmailToPDFConverter(EmailConverter):
    def __init__(
            self,
            template_dir: str = os.path.join(os.path.dirname(__file__), "../../utils/templates"),
            template_name: str = "EmailToPdfHtml_template.html"
    ) -> None:
        try:
            template_dir = os.path.abspath(template_dir)
            self.env: Environment = Environment(loader=FileSystemLoader(template_dir))
            self.template: Template = self.env.get_template(template_name)
            logger.info(f"Template loaded from: {template_dir}")
        except Exception as e:
            logger.exception("Failed to initialize Jinja2 environment or load template.")
            raise RuntimeError("Template loading failed") from e

    def convert_to_pdf(self, email_data: Dict[str, Any], output_path: str) -> str:
        """
        Convert structured email data into a PDF file.

        Args:
            email_data (dict): Email metadata and body (subject, sender, received_at, etc.).
            output_path (str): Target file path to save the resulting PDF.
        """
        try:
            assert isinstance(email_data, dict), "email_data must be a dictionary"
            assert isinstance(output_path, str), "output_path must be a string"

            html: str = self.template.render(
                subject=email_data.get("subject", ""),
                sender=email_data.get("sender", ""),
                received_at=email_data.get("received_at", ""),
                body=email_data.get("body", ""),
                attachments=email_data.get("attachments", [])
            )

            logger.debug(f"Rendered HTML preview (first 500 chars):\n{html[:500]}")


            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            pdfkit.from_string(html, output_path)
            logger.info(f"✅ PDF successfully saved at: {os.path.abspath(output_path)}")
            return output_path
        except AssertionError as ae:
            logger.error(f"Type assertion failed: {ae}")
            raise ValueError(f"Invalid input: {ae}") from ae

        except Exception as e:
            logger.exception("❌ Failed to convert email to PDF.")
            raise RuntimeError("Email to PDF conversion failed") from e
