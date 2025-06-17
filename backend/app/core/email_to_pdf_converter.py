from jinja2 import  Environment, PackageLoader
import pdfkit
import os
from abc import ABC, abstractmethod
from backend.utils.file_utils import FilePathUtils

class EmailConverter(ABC):
    @abstractmethod
    def convert_to_pdf(self, email_data, output_path):
        pass

class HTMLEmailToPDFConverter(EmailConverter):
    def __init__(self, package_name="backend", template_dir="utils/templates", template_name="EmailToPdfHtml_template.html"):
        self.env = Environment(
            loader=PackageLoader(package_name, template_dir)
        )
        self.template = self.env.get_template(template_name)


    def convert_to_pdf(self, email_data, output_path):
        # template = Template(self.TEMPLATE)
        html = self.template.render(
            subject=email_data["subject"],
            sender=email_data["sender"],
            received_at=email_data["received_at"],
            body=email_data.get("body", ""),
            attachments=email_data.get("attachments", [])
        )

        # Save as PDF
        try:

            file_utils = FilePathUtils(file=None, temp_dir=None)  #
            output_dir = file_utils.file_dir()

            # # Ensure output directory exists
            # output_dir = os.path.dirname(output_path)
            # os.makedirs(output_dir, exist_ok=True)

            # Optional: Print debug info
            print(f"Rendering HTML:\n{html[:500]}...")  # First 500 chars
            print(f"Saving PDF to: {os.path.abspath(output_dir)}")

            # Generate PDF
            pdfkit.from_string(html, output_dir)
            print("✅ PDF conversion completed.")
        except Exception as e:
            print(f"❌ Error converting to PDF: {e}")
            raise