# app/utils/email_to_pdf.py
from pathlib import Path
from typing import Union
from reportlab.pdfgen import canvas

class EmailToPdf:
    @staticmethod
    def generate_pdf_from_text(text_content: str, output_path: Union[str, Path]):
        """
        Generate a PDF from plain text content using pdfgen (ReportLab).
        """
        c = canvas.Canvas(str(output_path))
        text_lines = text_content.split('\n')

        # Set font and size
        c.setFont("Helvetica", 12)

        # Starting position
        x, y = 50, 750

        # Write each line
        for line in text_lines:
            c.drawString(x, y, line)
            y -= 15  # Move down for next line

            # Handle page overflow
            if y < 50:
                c.showPage()
                c.setFont("Helvetica", 12)
                y = 750

        c.save()
        return Path(output_path).resolve()