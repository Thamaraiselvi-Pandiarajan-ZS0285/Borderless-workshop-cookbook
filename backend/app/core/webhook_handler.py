# Use FastAPI or Flask for webhook handling
from fastapi import FastAPI, Request
from backend.app.core.email_processor import EmailProcessor

app = FastAPI()

@app.post("/email/webhook")
async def handle_webhook(request: Request):
    processor = EmailProcessor()
    payload = await request.json()
    email_data = extract_email_from_payload(payload)
    try:
        processor.process_emails(email_data)
        return {"status": "processed"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

def extract_email_from_payload(payload):
    # Implement for your email service (e.g., Microsoft Graph)
    return payload["email"]
