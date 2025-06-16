from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
import json
import os

# Import converters
from backend.utils.HTMLEmailToPDFConverter import HTMLEmailToPDFConverter
from backend.utils.HighResPDFToImageConverter import HighResPDFToImageConverter

app = FastAPI()

# Ensure output directories exist
os.makedirs("output", exist_ok=True)
os.makedirs("output/images", exist_ok=True)


@app.post("/convert/email-to-pdf")
async def convert_email_to_pdf(email_file: UploadFile = File(...)):
    try:
        contents = await email_file.read()
        email_data = json.loads(contents)

        base_name = os.path.splitext(email_file.filename)[0]
        pdf_path = f"output/{base_name}.pdf"

        pdf_converter = HTMLEmailToPDFConverter()
        pdf_converter.convert_to_pdf(email_data, pdf_path)

        return FileResponse(pdf_path, media_type='application/pdf', filename="converted_email.pdf")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing email: {e}")

@app.post("/convert/pdf-to-images")
async def convert_pdf_to_images(pdf_file: UploadFile = File(...),source_type:str = Form("unknown")):
    try:
        pdf_path = f"output/{pdf_file.filename}"
        with open(pdf_path, "wb") as buffer:
            buffer.write(await pdf_file.read())

        image_converter = HighResPDFToImageConverter()
        image_paths = image_converter.convert(pdf_path, f"output/images/{pdf_file.filename}", 500,source_type)

        # Return list of image paths
        return {
            "message": f"{len(image_paths)} pages converted to images at 500 DPI",
            "images": [f"output/images/{pdf_file.filename}/{os.path.basename(p)}" for p in image_paths],
            "source_type": source_type
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting PDF: {e}")

@app.get("/images/{image_name}")
async def get_image(image_name: str):
    image_path = f"output/images/{image_name}"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)

# For testing purposes only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)