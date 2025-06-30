import asyncio
import os
import uuid
from typing import Any, Dict, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.messages import HandoffMessage, BaseChatMessage
from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import logging
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from autogen_core.models import (
    UserMessage,
)

from backend.app.core.classifier_agent import EmailClassifierProcessor
from backend.app.core.email_to_pdf_converter import HTMLEmailToPDFConverter
from backend.app.core.embedder import Embedder
from backend.app.core.file_operations import FileToBase64
from backend.app.core.metadata_consolidator import MetadataConsolidatorAgent
from backend.app.core.metadata_validation import MetadataValidatorAgent
from backend.app.core.ocr_agent import EmailOCRAgent
from backend.app.core.paper_itemizer import PaperItemizer
from backend.app.core.summarization_agent import SummarizationAgent
from backend.app.request_handler.email_request import EmailClassifyImageRequest
from backend.app.request_handler.metadata_extraction import EmailImageRequest
from backend.app.request_handler.paper_itemizer import PaperItemizerRequest
from backend.app.response_handler.email_classifier_response import email_classify_response_via_vlm
from backend.app.response_handler.paper_itemizer import build_paper_itemizer_response
from backend.config.dev_config import DEFAULT_IMAGE_FORMAT
from backend.prompts.summarization_prompt import TASK_VARIANTS
from backend.utils.extract_data_from_file import AttachmentExtractor, split_into_pages


# ----------------------- Configure Logging -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EmailProcessingPipeline")


# ----------------------- Azure Config -----------------------
AZURE_OPENAI_API_KEY = "2K3oZXs28WZE2y1Fzg1jIPUdPGSY3xF0cWPcDx2DlF4RpUKimG0DJQQJ99BFACYeBjFXJ3w3AAABACOG9Osf"
AZURE_OPENAI_ENDPOINT = "https://bap-openai.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-41-mini"
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1-mini"

config_list = [
    {
        "model": AZURE_OPENAI_DEPLOYMENT,
        "api_type": "azure",
        "api_key": AZURE_OPENAI_API_KEY,
        "base_url": AZURE_OPENAI_ENDPOINT,
        "api_version": AZURE_OPENAI_API_VERSION,
    }
]
llm_config = {
    "config_list": config_list,
    "temperature": 0.1,
    "timeout": 120,
}

model_client = AzureOpenAIChatCompletionClient(
    model=AZURE_OPENAI_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_ad_token=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME
)





def convert_email_data_to_pdf(email_data: dict) -> dict:
    output_dir = "/tmp/email_pipeline_data"
    os.makedirs(output_dir, exist_ok=True)
    file_name = str(uuid.uuid4())
    pdf_path = os.path.join(output_dir, f"{file_name}.pdf")

    pdf_converter = HTMLEmailToPDFConverter()
    pdf_converter.convert_to_pdf(email_data, pdf_path)

    logger.info(f"✅ PDF generated at: {pdf_path}")
    return {"path": pdf_path, "file_name": file_name}


def do_encode_via_path(path: str, file_name: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found at {path}")

    base64_encoder = FileToBase64(str(path))
    encoded_data = base64_encoder.do_base64_encoding_by_file_path()

    return PaperItemizerRequest(
        input=encoded_data,
        file_name=file_name,
        file_extension=DEFAULT_IMAGE_FORMAT
    )


def do_paper_itemizer(data: Dict[str, Any]) -> Dict[str, Any]:
    req = PaperItemizerRequest(**data)
    result = PaperItemizer(input=req.input, file_name=req.file_name, extension=req.file_extension).do_paper_itemizer()
    return build_paper_itemizer_response(result, 200, "Paper itemization successful.")

# Define agents
planner = AssistantAgent(
    name="planner",
    model_client=model_client,
    handoffs=["email_processor", "pdf_encoder", "image_converter"],
    system_message="""You are a data Piepline processing agent.
    Your role is to:
    1. Initiate the workflow by handing off to the email_processor with the email data.
    2. Coordinate handoffs between agents based on their outputs.
    3. Terminate the workflow with 'TERMINATE' when the report is generated.
    Workflow:
    - Send email data to email_processor to convert to PDF.
    - Pass PDF path and file name to pdf_encoder for base64 encoding.
    - Pass encoded data to image_converter for report generation.
    - Terminate after receiving the final report.
    Keep track of the workflow state and ensure proper handoffs.""",
)

email_processor = AssistantAgent(
    name="email_processor",
    model_client=model_client,
    handoffs=["planner"],
    tools=[convert_email_data_to_pdf],
    system_message="""You are an email processor.
    Use the convert_email_data_to_pdf tool to convert email data to a PDF.
    Return the result (PDF path and file name) to the planner.""",
)

pdf_encoder = AssistantAgent(
    name="pdf_encoder",
    model_client=model_client,
    handoffs=["planner"],
    tools=[do_encode_via_path],
    system_message="""You are a PDF encoder.
    Use the do_encode_via_path tool to encode the PDF to base64.
    Return the encoded data (PaperItemizerRequest) to the planner.""",
)

report_generator = AssistantAgent(
    name="image_converter",
    model_client=model_client,
    handoffs=["planner"],
    tools=[do_paper_itemizer],
    system_message="""You are a image converter.
    Use the do_paper_itemizer tool to generate a report from the encoded data.
    Return the report to the planner.""",
)

# Define termination condition
text_termination = TextMentionTermination("TERMINATE")
termination = text_termination

# Create Swarm
pipeline_executor_team = Swarm(
    participants=[planner, email_processor, pdf_encoder, report_generator],
    termination_condition=termination
)

# Run the task

email_data = {
        "sender": "koumiya",
        "subject": "Q2 Financial Report Submission",
        "received_at": "2025-06-19T09:40:11.106Z",
        "body": """Dear Akhil,
    We are delighted to inform you that Borderless Access has been selected as the awarded vendor for the Multi-Country Healthcare Market Research project.
    Your proposal demonstrated exceptional understanding of the healthcare domain.
    Our operations team will reach out within the next week to schedule a formal project kickoff on July 16, 2025.

    Regards,
    Radhika Mehta
    Insights Lead
    HealthWorld Insights Consortium""",
        "hasAttachments": False,
        "attachments": []
    }


async def run_team_stream() -> None:
    task_result = await Console(pipeline_executor_team.run_stream(task=str(email_data)))
    last_message = task_result.messages[-1]

    while isinstance(last_message, HandoffMessage) and last_message.target == "user":
        user_message = input("User: ")

        task_result = await Console(
            pipeline_executor_team.run_stream(task=HandoffMessage(source="user", target=last_message.source, content=user_message))
        )
        last_message = task_result.messages[-1]

    await run_team_stream()
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(run_team_stream())