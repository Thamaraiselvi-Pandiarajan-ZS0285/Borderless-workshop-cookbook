from openai import AzureOpenAI
from backend.config.db_config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION
)


class EmailOCRAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION
        )
        self.model = AZURE_OPENAI_DEPLOYMENT

    def extract_text_from_base64(self, base64_str: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise email OCR agent. Your role is to extract all visible information from an image of an email. "
                            "The image will contain an official or business email in screenshot or scanned format.\n\n"
                            "Your job is to analyze the entire content of the image and extract all the text exactly as it appears — including subject, from, to, date, and the full body of the email. "
                            "Do not summarize. Do not omit or ignore lines, headers, or paragraphs.\n\n"
                            "You must extract and return everything visible in the image in the following structured JSON format:\n\n"
                            "From: [Extracted sender name or email, if visible]\n"
                            "To: [Recipient(s), if visible]\n"
                            "Date: [Full date and time, if shown]\n"
                            "Subject: [Full subject line]\n"
                            "Body:\n"
                            "[Full body text exactly as shown in the image — all lines must be included with correct line breaks and formatting]\n\n"
                            "Signature:\n"
                            "[Extract name, title, company, and any footer or disclaimer text]\n\n"
                            "Attachments:\n"
                            "[If the image mentions any attachments, list them here]\n\n"
                            "Other Info:\n"
                            "[Any other visible text, like legal disclaimers, quoted replies, or thread context]\n\n"
                            "⚠️ Rules:\n"
                            "- Do NOT leave out any line of the body.\n"
                            "- Do NOT skip headers, footers, or sign-offs.\n"
                            "- Return all visible text — even if blurry or partial.\n"
                            "- Do NOT infer, summarize, or guess — extract only what's visible.\n\n"
                            "Example Output:\n"
                            "From: John Doe <john@example.com>\n"
                            "To: vendor@example.com\n"
                            "Date: Jan 1, 2025\n"
                            "Subject: Award of Contract – CRM Solution Implementation\n"
                            "Body:\n"
                            "Dear [Vendor Name],\n\n"
                            "[...full message body here, line-by-line...]\n\n"
                            "Signature:\n"
                            "John Doe\n"
                            "Director – IT Projects\n"
                            "GlobalCo Ltd.\n\n"
                            "Attachments:\n"
                            "Letter of Intent (LOI)\n\n"
                            "Other Info:\n"
                            "N/A"
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_str}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise RuntimeError(f"OCR Extraction Failed: {e}")
