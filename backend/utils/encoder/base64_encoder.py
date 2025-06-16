# app/utils/encoder/base64_encoder.py
import base64
from pathlib import Path

def encode_pdf_to_base64(pdf_path: Path) -> str:
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")