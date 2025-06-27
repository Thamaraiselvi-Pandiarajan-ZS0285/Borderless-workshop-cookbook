import logging

from typing import Optional

from backend.src.core.base_agents.base_agent import BaseAgent
from backend.src.core.base_client.base_client import OpenAiClient
from backend.src.prompts.meta_data_extraction import RFP_EXTRACTION_PROMPT, BID_WIN_EXTRACTION_PROMPT, \
    BID_REJECTION_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


def get_prompt_by_category(category: str) -> str:
    match category.lower():
        case "rfp":
            return RFP_EXTRACTION_PROMPT
        case "bid-win":
            return BID_WIN_EXTRACTION_PROMPT
        case "rejection":
            return BID_REJECTION_EXTRACTION_PROMPT
        case _:
            raise ValueError(f"Unsupported category: {category}")


class EmailOCRAgent:
    """
    An agent that uses Azure OpenAI Vision models to extract metadata or text from base64-encoded email images.
    """

    def __init__(self) -> None:
            self.client = OpenAiClient().open_ai_chat_completion_client
            self.base_agent=BaseAgent(self.client)


    async def extract_text_from_base64(self, base64_str: str, category: str) -> str:
        """
        Extracts structured content or metadata from a base64-encoded image using a vision-enabled OpenAI model.

        Args:
            base64_str (str): The base64 string of the email image.

        Returns:
            str: Extracted content from the image.

        Raises:
            RuntimeError: If the API call fails or returns no content.
            :param base64_str:
            :param category:
        """
        if not base64_str or not isinstance(base64_str, str):
            raise ValueError("Input base64 string is invalid or empty.")

        prompt = get_prompt_by_category(category)
        ocr_agent = self.base_agent.create_assistant_agent("OCR_AGENT", prompt)

        content = (
            f"Image URL:\n"
            f"Type: image_url\n"
            f"Detail: high\n"
            f"URL: data:image/jpeg;base64,{base64_str}"
        )

        try:
            response = await ocr_agent.run(task=content)
            extracted_text: Optional[str] = response.choices[0].message.content
            if not extracted_text:
                logger.warning("⚠️ No text was returned by the OCR model.")
                raise RuntimeError("No content extracted from image.")

            return extracted_text.strip()

        except Exception as e:
            logger.exception("❌ OCR extraction via Azure OpenAI failed.")
            raise RuntimeError(f"OCR Extraction Failed: {e}") from e



