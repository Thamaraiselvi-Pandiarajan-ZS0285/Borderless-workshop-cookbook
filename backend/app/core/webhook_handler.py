# Use FastAPI or Flask for webhook handling
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
import json
import asyncio
from backend.app.core.email_processor import EmailProcessor
from backend.config.db_config import *
from backend.db.db_helper.db_Initializer import DbInitializer

app = FastAPI()

@app.post("/email/webhook")
async def handle_webhook(request: Request):
    db_init = DbInitializer(
        POSTGRESQL_DRIVER_NAME, POSTGRESQL_HOST, POSTGRESQL_DB_NAME,
        POSTGRESQL_USER_NAME, POSTGRESQL_PASSWORD, POSTGRESQL_PORT_NO
    )

    app.state.db_engine = db_init.db_create_engin()
    app.state.db_session = db_init.db_create_session()

    processor = EmailProcessor(app.state.db_session)
    try:
        payload = extract_email_from_payload(request)
        notifications = payload.get("value", [])

        for notification in notifications:
            resource_data = notification.get("resourceData", {})
            message_id = resource_data.get("id")
            if not message_id:
                continue

            # EmailProcessor to process the email
            print(f"[Webhook] New email triggered with ID: {message_id}")
            await asyncio.to_thread(processor.process_email_by_id, message_id)

        return {"status": "processed"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

# Microsoft Graph email service
async def extract_email_from_payload(request: Request):
    # Step 1: Microsoft Graph subscription validation (echo validationToken)
    if "validationToken" in request.query_params:
        token = request.query_params["validationToken"]
        return PlainTextResponse(content=token, status_code=200)

    # Step 2: Actual notification (message created)
    body = await request.body()

    return json.loads(body)