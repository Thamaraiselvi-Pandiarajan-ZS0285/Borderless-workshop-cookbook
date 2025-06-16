# app/utils/email_to_pdf.py
from io import BytesIO
from reportlab.pdfgen import canvas


class EmailToPdf:
    @staticmethod
    def generate_pdf_from_text(graph_email: dict) -> bytes:
        """
        Convert Microsoft Graph API email JSON directly into PDF bytes.
        No intermediate .eml file or HTML needed.
        """
        buffer = BytesIO()
        c = canvas.Canvas(buffer)
        c.setFont("Helvetica-Bold", 14)

        # Extract email data from Graph API response
        subject = graph_email.get("subject", "No Subject")
        from_data = graph_email.get("from", {}).get("emailAddress", {})
        from_name = from_data.get("name", "Unknown")
        from_email = from_data.get("address", "unknown@example.com")
        to_recipients = ", ".join([
            recipient.get("emailAddress", {}).get("address", "")
            for recipient in graph_email.get("toRecipients", [])
        ]) or "Unknown"
        received = graph_email.get("receivedDateTime", "Unknown")
        body_html = graph_email.get("body", {}).get("content", "<p>No content available.</p>")

        # Strip HTML tags for plain text rendering
        import re
        body_text = re.sub('<[^<]+?>', '', body_html).strip()
        body_lines = body_text.splitlines() or ["(No readable content)"]

        # Write headers
        c.drawString(50, 800, f"Subject: {subject}")
        c.drawString(50, 780, f"From: {from_name} <{from_email}>")
        c.drawString(50, 760, f"To: {to_recipients}")
        c.drawString(50, 740, f"Date: {received}")

        c.line(50, 730, 550, 730)

        # Body
        c.setFont("Helvetica", 12)
        y = 710
        for line in body_lines:
            if y < 50:
                c.showPage()
                y = 750
            c.drawString(50, y, line.strip())
            y -= 15

        c.save()

        return buffer.getvalue()