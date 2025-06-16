from fastapi import FastAPI, HTTPException
from models import EmailInput
from utils.email_to_pdf import EmailToPdf
from utils.parser.eml_parser import parse_eml_content
from utils.parser.html_parser import parse_html_content
from utils.parser.xml_parser import parse_xml_content
from utils.encoder.base64_encoder import Encoder


app = FastAPI()
pdf_generator = EmailToPdf()
encoder = Encoder()

@app.post("/generate-pdf/")
def generate_pdf(data: EmailInput):
    try:
        # Step 1: Parse input format (eml/html/xml) into email_data
        if data.format == "eml":
            email_data = parse_eml_content(data.content)
        elif data.format == "html":

            email_data = parse_html_content(data.content)
        elif data.format == "xml":

            email_data = parse_xml_content(data.content)
        else:
            raise ValueError(f"Unsupported format: {data.format}")

        # Step 2: Generate PDF in memory
        pdf_bytes = pdf_generator.generate_pdf_from_text(email_data)

        # Step 3: Encode as Base64
        base64_pdf = encoder.encode_pdf_to_base64(pdf_bytes)

        return {"pdf_base64": base64_pdf}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))