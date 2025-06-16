from fastapi import FastAPI, HTTPException
from models import EmailInput
from fastapi.responses import FileResponse
from utils.email_to_pdf import EmailToPdf


app = FastAPI()
pdf_generator = EmailToPdf()

@app.post("/generate-pdf/")
def generate_pdf(input_data: EmailInput):
    try:
        pdf_path = pdf_generator.generate_pdf_from_text(input_data.email)
        return FileResponse(pdf_path, media_type='application/pdf', filename=pdf_path.split("/")[-1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
