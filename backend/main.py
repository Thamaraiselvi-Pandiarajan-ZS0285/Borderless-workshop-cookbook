from fastapi import FastAPI
from app.classifier_agent import classify_email
from models.emailRequest import EmailRequest


app = FastAPI()

@app.post("/classify_email")
def classify(email: EmailRequest) -> dict:
    result = classify_email(email.subject, email.body)
    return result