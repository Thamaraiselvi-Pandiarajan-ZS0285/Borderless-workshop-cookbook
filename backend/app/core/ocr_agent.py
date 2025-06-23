import logging
from openai import AzureOpenAI
from typing import Optional

from backend.app.request_handler.metadata_extraction import EmailImageRequest
from backend.prompts.meta_data_extraction import RFP_EXTRACTION_PROMPT,BID_WIN_EXTRACTION_PROMPT, BID_REJECTION_EXTRACTION_PROMPT

# Assume these constants are imported from your config
from backend.config.dev_config import *

logger = logging.getLogger(__name__)


class EmailOCRAgent:
    """
    An agent that uses Azure OpenAI Vision models to extract metadata or text from base64-encoded email images.
    """

    def __init__(self) -> None:
        try:
            self.client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_API_VERSION
            )
            self.model = AZURE_OPENAI_DEPLOYMENT_NAME
            logger.info("✅ Azure OpenAI client initialized successfully.")
        except Exception as e:
            logger.exception("❌ Failed to initialize Azure OpenAI client.")
            raise RuntimeError(f"Initialization Failed: {e}") from e

    def get_prompt_by_category(self, category: str) -> str:
        match category.lower():
            case "rfp":
                return RFP_EXTRACTION_PROMPT
            case "bid-win":
                return BID_WIN_EXTRACTION_PROMPT
            case "rejection":
                return BID_REJECTION_EXTRACTION_PROMPT
            case _:
                raise ValueError(f"Unsupported category: {category}")

    def extract_text_from_base64(self, base64_str: str, category: str) -> str:
        """
        Extracts structured content or metadata from a base64-encoded image using a vision-enabled OpenAI model.

        Args:
            base64_str (str): The base64 string of the email image.

        Returns:
            str: Extracted content from the image.

        Raises:
            RuntimeError: If the API call fails or returns no content.
        """
        if not base64_str or not isinstance(base64_str, str):
            raise ValueError("Input base64 string is invalid or empty.")

        prompt = self.get_prompt_by_category(category)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt
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

            extracted_text: Optional[str] = response.choices[0].message.content
            if not extracted_text:
                logger.warning("⚠️ No text was returned by the OCR model.")
                raise RuntimeError("No content extracted from image.")

            return extracted_text.strip()

        except Exception as e:
            logger.exception("❌ OCR extraction via Azure OpenAI failed.")
            raise RuntimeError(f"OCR Extraction Failed: {e}") from e



