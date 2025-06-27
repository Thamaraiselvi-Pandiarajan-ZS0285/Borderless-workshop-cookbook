import asyncio
import base64
from operator import truediv
from typing import Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from openai import AzureOpenAI
from pdf2image import convert_from_path

from backend.app.core.file_operations import FileToBase64

AZURE_OPENAI_API_KEY="2K3oZXs28WZE2y1Fzg1jIPUdPGSY3xF0cWPcDx2DlF4RpUKimG0DJQQJ99BFACYeBjFXJ3w3AAABACOG9Osf"
AZURE_OPENAI_ENDPOINT= "https://bap-openai.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-41-mini"
AZURE_OPENAI_API_VERSION="2025-01-01-preview"
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1-mini"
MODEL_INFO={
                "family": "gpt-4",
                "vision": True,
                "structured_output": False,
                "function_calling": False,
                "json_output": False
            }


CLASSIFICATION_PROMPT_VLM = """
   You are an email classification assistant for a market research firm. Your task is to analyze the **body** of incoming business emails and classify each email into one of the following categories.

        Input:
        - The email is provided as an image in base64-encoded format. First, **extract the text content using OCR** (preferably high-accuracy OCR such as Tesseract or Azure OCR).
        - The extracted content may include plain text, HTML fragments, or scanned documents.
        - If an attachment is included, summarize its content to help support the classification if needed.

        Categories:
        - **"RFP"**: Indicates a Request for Proposal, quotation, or invitation to bid. Typical language: "request for proposal", "quotation", "pricing request", "invitation to bid", "scope of work".
        - **"Bid-Win"**: Indicates the proposal was accepted. Look for language such as: "your proposal has been selected", "you have been awarded", "we are pleased to inform you", "congratulations".
        - **"Rejection"**: Indicates the proposal was not selected. Typical phrases include: "we regret to inform you", "not selected", "unfortunately", "thank you for your submission".
        - **"Unclear"**: Use this if the content does not provide enough information to determine a clear category.

        Guidelines:
        1. Begin by accurately extracting and preprocessing the email body text from the base64-encoded image.
        2. Focus on **semantic meaning, intent, and tone** of the message—**not just keyword matching**.
        3. Prioritize the **main message content**. Ignore headers, footers, and email signatures unless they convey relevant intent.
        4. If the body lacks clarity or decisive content, refer to any **attachment summary** for additional context.
        5. Always return exactly **one of the following labels**: `"RFP"`, `"Bid-Win"`, `"Rejection"`, or `"Unclear"`.
        6. Do **not** provide any explanation, justification, or extra output—only the classification label.

        Output Format:
        RFP | Bid-Win | Rejection | Unclear
"""


def convert_pdf_to_jpeg(pdf_path: str, output_path: str = "temp_image1.jpg") -> str:
    pages = convert_from_path(pdf_path, first_page=1, last_page=1)
    pages[0].save(output_path, "JPEG")
    return output_path

# Helper: Encode image to base64
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

async def classify_image() -> str:

    file_path="/home/koumiya.subramani@zucisystems.com/Borderless_access/Borderless-workshop-cookbook-and-pilot/output/sample_2_fin 1.pdf"

    image_path = convert_pdf_to_jpeg(file_path)

    base64_image = encode_image_to_base64(image_path)

    if not base64_image or not isinstance(base64_image, str):
        raise ValueError("Invalid base64 input.")

    model_client = AzureOpenAIChatCompletionClient(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        model_info=MODEL_INFO
    )
    agent = AssistantAgent(
        name="vlm_classifier",
        system_message=CLASSIFICATION_PROMPT_VLM,
        model_client=model_client
    )

    # Compose the base64 string as plain text input for VLM
    task = f"""
    Below is the base64-encoded image of an email. Classify its content as per the instructions.

    Image (base64-encoded JPEG):
    data:image/jpeg;base64,{base64_image}
    """

    try:
        result = await agent.run(task=task)
        # Get the assistant's message (last one typically)
        assistant_reply = next(
            (msg.content for msg in result.messages if msg.source == 'vlm_classifier'),
            None
        )
        return assistant_reply

    except Exception as e:
        raise RuntimeError(f"OCR Extraction Failed: {e}") from e


if __name__ == "__main__":
    result = asyncio.run(classify_image())
    print("Extracted Text:", result)
