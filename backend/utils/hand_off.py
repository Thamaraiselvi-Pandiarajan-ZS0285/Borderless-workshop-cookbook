import logging
import os
import json
import re
import uuid
from typing import List, Dict, Any

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core import (
    FunctionCall,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool, Tool
from pydantic import BaseModel

# Backend modules
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

# ----------------------- Utility Functions -----------------------
def serialize_message_context(context: list):
    return [
        {
            "type": msg.type,
            "content": msg.content,
            "source": msg.source,
            "thought": getattr(msg, 'thought', None)
        } for msg in context
    ]

def convert_email_data_to_pdf(sender: str, subject: str, body: str, attachments: list) -> dict:
    output_dir = "/tmp/email_pipeline_data"
    os.makedirs(output_dir, exist_ok=True)
    file_name = str(uuid.uuid4())
    pdf_path = os.path.join(output_dir, f"{file_name}.pdf")

    email_data = {
        "sender": sender,
        "subject": subject,
        "body": body,
        "attachments": attachments
    }
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

def do_classify_via_vlm(data: Dict[str, Any]) -> Dict[str, Any]:
    req = EmailClassifyImageRequest(**data)
    classifier = EmailClassifierProcessor()
    summarizer = SummarizationAgent()
    extractor = AttachmentExtractor()

    full = req.json_data.body
    attach_summary = ""

    if req.json_data.hasAttachments:
        pages = split_into_pages(extractor.extract_many(req.json_data.attachments))
        page_summaries = [f"Page {i + 1}:\n{summarizer.summarize_text(p)}" for i, p in enumerate(pages)]
        attach_summary = summarizer.summarize_text("\n".join(page_summaries))
        full += "\nAttachment Summary\n\n" + attach_summary

    extracted = []
    for item in req.imagedata:
        base64_img = (
            FileToBase64(item.input_path).do_base64_encoding_by_file_path()
            if os.path.exists(item.input_path) else item.input_path
        )
        extracted.append(classifier.classify_via_vlm(base64_img))

    summary = "".join(
        f"{label}:\n{summarizer.summarize_text(full, variant)}\n\n"
        for label, variant in TASK_VARIANTS.items()
    )

    return email_classify_response_via_vlm(req, extracted, summary)

def build_email_image_request(data: Dict[str, Any]) -> Dict[str, Any]:
    resp = data["paper_itemizer_response"]
    classification = data["classification_result"]
    return {
        "data": [
            {
                "input": r["filePath"],
                "file_name": r["fileName"],
                "file_extension": r["fileExtension"],
                "category": classification["classification"],
            } for r in resp["results"]
        ]
    }

def upload_email_images(data: Dict[str, Any]) -> Dict[str, Any]:
    req = EmailImageRequest(**data)
    ocr = EmailOCRAgent()
    val = MetadataValidatorAgent()
    cons = MetadataConsolidatorAgent()

    extracted, errors = [], []

    for item in req.data:
        try:
            base64_img = FileToBase64(item.input).do_base64_encoding_by_file_path()
            cleaned = re.sub(
                r"^```json\s*|\s*```$", "",
                ocr.extract_text_from_base64(base64_img, item.category).strip()
            )
            val.validate_metadata(cleaned, item.category)
            extracted.append(cleaned)
        except Exception as e:
            errors.append({"file_name": item.file_name, "error": str(e)})

    if errors:
        return {"errors": errors}

    return {"consolidated_metadata": cons.consolidate(extracted, category=req.data[0].category)}

def ingest_all_embeddings(data: Dict[str, Any]) -> Dict[str, Any]:
    resp = data["upload_response"]
    summary = data["classification_summary"]
    for r in resp.get("results", []):
        text = (
            f"Subject: {r['extracted_metadata']['subject']}\n\n"
            f"{r['extracted_metadata']['full_email_text']}\nAttachment Summary:{summary}"
        )
        Embedder(None, None).ingest_email_for_content(text, text)
    return {"status": "success"}

# ----------------------- Agent Definitions -----------------------

class UserLogin(BaseModel):
    pass

class UserTask(BaseModel):
    context: List[LLMMessage]

class AgentResponse(BaseModel):
    reply_to_topic_type: str
    context: List[dict]

class AIAgent(RoutedAgent):
    def __init__(
        self,
        description: str,
        system_message: SystemMessage,
        model_client: ChatCompletionClient,
        tools: List[Tool],
        delegate_tools: List[Tool],
        agent_topic_type: str,
        user_topic_type: str,
    ) -> None:
        super().__init__(description)
        self._system_message = system_message
        self._model_client = model_client
        self._tools = dict([(tool.name, tool) for tool in tools])
        self._tool_schema = [tool.schema for tool in tools]
        self._delegate_tools = dict([(tool.name, tool) for tool in delegate_tools])
        self._delegate_tool_schema = [tool.schema for tool in delegate_tools]
        self._agent_topic_type = agent_topic_type
        self._user_topic_type = user_topic_type

    @message_handler
    async def handle_task(self, message: UserTask, ctx: MessageContext) -> None:
        logger.info(f"[Agent:{self.id.type}] Handling task with context: {message.context}")
        try:
            llm_result = await self._model_client.create(
                messages=[self._system_message] + message.context,
                tools=self._tool_schema + self._delegate_tool_schema,
                cancellation_token=ctx.cancellation_token,
            )
            logger.info(f"[Agent:{self.id.type}] LLM Response: {llm_result.content}")

            while isinstance(llm_result.content, list) and all(isinstance(m, FunctionCall) for m in llm_result.content):
                tool_call_results = []
                delegate_targets = []

                for call in llm_result.content:
                    logger.info(
                        f"[Agent:{self.id.type}] Processing FunctionCall: {call.name} with arguments {call.arguments}")
                    arguments = json.loads(call.arguments)

                    if call.name in self._tools:
                        logger.info(f"[Agent:{self.id.type}] Running tool: {call.name}")
                        result = await self._tools[call.name].run_json(arguments, ctx.cancellation_token)
                        logger.info(f"[Agent:{self.id.type}] Tool {call.name} output: {result}")
                        result_as_str = self._tools[call.name].return_value_as_string(result)
                        tool_call_results.append(
                            FunctionExecutionResult(call_id=call.id, content=result_as_str, is_error=False,
                                                    name=call.name))

                    elif call.name in self._delegate_tools:
                        logger.info(f"[Agent:{self.id.type}] Delegating to tool: {call.name}")
                        result = await self._delegate_tools[call.name].run_json(arguments, ctx.cancellation_token)
                        topic_type = self._delegate_tools[call.name].return_value_as_string(result)
                        logger.info(f"[Agent:{self.id.type}] Delegating to topic: {topic_type}")
                        delegate_messages = list(message.context) + [
                            AssistantMessage(content=[call], source=self.id.type),
                            FunctionExecutionResultMessage(
                                content=[
                                    FunctionExecutionResult(
                                        call_id=call.id,
                                        content=f"Transferred to {topic_type}. Adopt persona immediately.",
                                        is_error=False,
                                        name=call.name,
                                    )
                                ]
                            ),
                        ]
                        delegate_targets.append((topic_type, UserTask(context=delegate_messages)))
                    else:
                        logger.error(f"[Agent:{self.id.type}] Unknown tool called: {call.name}")
                        raise ValueError(f"Unknown tool: {call.name}")

                if delegate_targets:
                    for topic_type, task in delegate_targets:
                        logger.info(f"[Agent:{self.id.type}] Publishing to delegate topic: {topic_type}")
                        await self.publish_message(task, topic_id=TopicId(topic_type, source=self.id.key))

                if tool_call_results:
                    message.context.extend(
                        [
                            AssistantMessage(content=llm_result.content, source=self.id.type),
                            FunctionExecutionResultMessage(content=tool_call_results),
                        ]
                    )
                    llm_result = await self._model_client.create(
                        messages=[self._system_message] + message.context,
                        tools=self._tool_schema + self._delegate_tool_schema,
                        cancellation_token=ctx.cancellation_token,
                    )
                    logger.info(f"[Agent:{self.id.type}] LLM Response after tool execution: {llm_result.content}")
                else:
                    logger.info(f"[Agent:{self.id.type}] Task delegated, exiting.")
                    return

            assert isinstance(llm_result.content, str)
            message.context.append(AssistantMessage(content=llm_result.content, source=self.id.type))
            await self.publish_message(
                AgentResponse(
                    context=serialize_message_context(message.context),
                    reply_to_topic_type=self._agent_topic_type
                ),
                topic_id=TopicId(self._user_topic_type, source=self.id.key),
            )
            logger.info(f"[Agent:{self.id.type}] Final response published to user.")
        except Exception as e:
            logger.error(f"[Agent:{self.id.type}] Error during handle_task: {e}", exc_info=True)
            raise


class OutputAgent(RoutedAgent):
    def __init__(self, description: str) -> None:
        super().__init__(description)

    @message_handler
    async def handle_task_result(self, message: AgentResponse, ctx: MessageContext) -> None:
        logger.info(f"[OutputAgent] Received final pipeline result: {message.context[-1]}")
        print(f"{'-' * 80}\nPipeline completed successfully. Final result: {message.context[-1]['content']}")

# ----------------------- Tool Definitions -----------------------
convert_email_into_pdf = FunctionTool(
    convert_email_data_to_pdf,
    description="Convert email data to PDF. Arguments: sender (str), subject (str), body (str), attachments (list)."
)
encode_pdf = FunctionTool(do_encode_via_path, description="Encode the PDF file")
paper_itemizer = FunctionTool(do_paper_itemizer, description="Extracts structured content from PDF")
classify_via_llm = FunctionTool(do_classify_via_vlm, description="Classifies email content")
email_image_request = FunctionTool(build_email_image_request, description="Builds image processing request")
extract_images = FunctionTool(upload_email_images, description="Performs OCR and metadata extraction")
ingest_embeddings = FunctionTool(ingest_all_embeddings, description="Ingests embeddings")

# ----------------------- Delegation Tool Functions -----------------------
def move_to_convert_email_to_pdf() -> str:
    return "ConvertEmailToPdf"

def move_to_encode_agent() -> str:
    return "FileEncoderAgent"

def move_to_paper_itemizer() -> str:
    return "PaperItemizerAgent"

def move_to_classify_via_llm() -> str:
    return "ClassifyViaLlmAgent"

def move_to_email_image_request() -> str:
    return "EmailImageRequestAgent"

def move_to_extract_images() -> str:
    return "ExtractImageAgent"

def move_to_ingest_embeddings() -> str:
    return "IngestEmbeddingAgent"

def move_back_to_triage() -> str:
    return "PipelineExecutorAgent"

def escalate_to_human_agent() -> str:
    return "HumanAgent"

def transfer_back_to_pipeline_execution_agent() -> str:
    return "PipelineExecutorAgent"

transfer_to_email_to_pdf = FunctionTool(
    move_to_convert_email_to_pdf,
    description="Move to PDF conversion"
)
transfer_to_encode_agent = FunctionTool(
    move_to_encode_agent,
    description="Move to encoding"
)
transfer_to_paper_itemizer = FunctionTool(
    move_to_paper_itemizer,
    description="Move to paper itemizer"
)
transfer_to_classify_via_llm = FunctionTool(
    move_to_classify_via_llm,
    description="Move to classification"
)
transfer_to_email_image_request = FunctionTool(
    move_to_email_image_request,
    description="Move to image request"
)
transfer_to_extract_images = FunctionTool(
    move_to_extract_images,
    description="Move to image extraction"
)
transfer_to_ingest_embeddings = FunctionTool(
    move_to_ingest_embeddings,
    description="Move to embedding ingestion"
)

# ---------------- Topic Types (Agent Names) ----------------
pipeline_executor_agent_topic_type = "PipelineExecutorAgent"
convert_email_agent_name = "ConvertEmailToPdf"
encode_agent_topic_type = "FileEncoderAgent"
paper_itemizer_topic_type = "PaperItemizerAgent"
classify_via_llm_topic_type = "ClassifyViaLlmAgent"
email_image_request_topic_type = "EmailImageRequestAgent"
extract_images_topic_type = "ExtractImageAgent"
ingest_embeddings_topic_type = "IngestEmbeddingAgent"
human_agent_topic_type = "HumanAgent"
output_agent_topic_type = "OutputAgent"

async def do_create_agent():
    runtime = SingleThreadedAgentRuntime()

    # Register OutputAgent
    output_agent_type = await OutputAgent.register(
        runtime,
        type=output_agent_topic_type,
        factory=lambda: OutputAgent(description="Agent to collect final pipeline output.")
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type=output_agent_topic_type, agent_type=output_agent_type.type)
    )

    # Register PipelineExecutorAgent
    executor_agent_type = await AIAgent.register(
        runtime,
        type=pipeline_executor_agent_topic_type,
        factory=lambda: AIAgent(
            description="Pipeline executor agent to start email processing.",
            system_message=SystemMessage(
                content="You are the pipeline executor agent for ACME Inc. Your job is to receive email data and delegate processing to the ConvertEmailToPdf agent. Use the transfer_to_email_to_pdf tool with the email data to start the pipeline."
            ),
            model_client=model_client,
            tools=[],
            delegate_tools=[
                transfer_to_email_to_pdf,
                transfer_to_encode_agent,
                transfer_to_paper_itemizer,
                transfer_to_classify_via_llm,
                transfer_to_email_image_request,
                transfer_to_extract_images,
                transfer_to_ingest_embeddings
            ],
            agent_topic_type=pipeline_executor_agent_topic_type,
            user_topic_type=output_agent_topic_type
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type=pipeline_executor_agent_topic_type, agent_type=executor_agent_type.type)
    )

    # Register ConvertEmailToPdf
    convert_email_agent_type = await AIAgent.register(
        runtime,
        type=convert_email_agent_name,
        factory=lambda: AIAgent(
            description="Agent to convert email content into PDF.",
            system_message=SystemMessage(
                content="""Convert raw email data into a PDF file.
    Input: Email data with sender, subject, body, and attachments.
    Use the convert_email_data_to_pdf tool with arguments: sender, subject, body, attachments.
    Output: { "path": pdf_path, "file_name": file_name }
    After conversion, delegate to FileEncoderAgent using transfer_to_encode_agent."""
            ),
            model_client=model_client,
            tools=[convert_email_into_pdf],
            delegate_tools=[transfer_to_encode_agent],
            agent_topic_type=convert_email_agent_name,
            user_topic_type=pipeline_executor_agent_topic_type,
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type=convert_email_agent_name, agent_type=convert_email_agent_type.type)
    )

    # Register FileEncoderAgent
    encode_agent_type = await AIAgent.register(
        runtime,
        type=encode_agent_topic_type,
        factory=lambda: AIAgent(
            description="Agent to encode the PDF into base64.",
            system_message=SystemMessage(
                content="""Encode a PDF file to base64.
    Input: 'path' (str), 'file_name' (str).
    Output: input: str
    file_name: str 
    file_extension: str = ".jpg"
    this would be the output of the encode agent and that to be transferred to be
    After encoding, delegate to PaperItemizerAgent using transfer_to_paper_itemizer."""
            ),
            model_client=model_client,
            tools=[encode_pdf],
            delegate_tools=[transfer_to_paper_itemizer],
            agent_topic_type=encode_agent_topic_type,
            user_topic_type=convert_email_agent_name,
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type=encode_agent_topic_type, agent_type=encode_agent_type.type)
    )

    # Register PaperItemizerAgent
    paper_itemizer_agent_type = await AIAgent.register(
        runtime,
        type=paper_itemizer_topic_type,
        factory=lambda: AIAgent(
            description="Agent to extract structured data from PDF.",
            system_message=SystemMessage(
                content="""You are the PaperItemizerAgent. Your task is to extract structured data from a PDF document by splitting it into image chunks (usually per page).
                Input: Extract the output from FileEncoderAgent's do_encode_via_path tool, available in the context as a FunctionExecutionResultMessage. This output is a dictionary with 'input' (base64 string), 'file_name' (string), and 'file_extension' (string, must be '.jpg').
                Action: Call the do_paper_itemizer tool with a argument containing these fields (e.g., {"input": "<base64>", "file_name": "<name>", "file_extension": ".jpg"}).
                Output: List of image file paths generated from the PDF.
                After processing, delegate to ClassifyViaLlmAgent using transfer_to_classify_via_llm."""
            ),
            model_client=model_client,
            tools=[paper_itemizer],
            delegate_tools=[transfer_to_classify_via_llm],
            agent_topic_type=paper_itemizer_topic_type,
            user_topic_type=encode_agent_topic_type,
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type=paper_itemizer_topic_type, agent_type=paper_itemizer_agent_type.type)
    )

    # Register ClassifyViaLlmAgent
    classify_agent_type = await AIAgent.register(
        runtime,
        type=classify_via_llm_topic_type,
        factory=lambda: AIAgent(
            description="Agent to classify the email content using LLM.",
            system_message=SystemMessage(
                content="""Classify email data into predefined categories.
    Input: structured extracted data.
    Output: Classification label with metadata.
    After classification, delegate to EmailImageRequestAgent using transfer_to_email_image_request."""
            ),
            model_client=model_client,
            tools=[classify_via_llm],
            delegate_tools=[transfer_to_email_image_request],
            agent_topic_type=classify_via_llm_topic_type,
            user_topic_type=paper_itemizer_topic_type,
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type=classify_via_llm_topic_type, agent_type=classify_agent_type.type)
    )

    # Register EmailImageRequestAgent
    email_image_request_agent_type = await AIAgent.register(
        runtime,
        type=email_image_request_topic_type,
        factory=lambda: AIAgent(
            description="Agent to prepare image processing request.",
            system_message=SystemMessage(
                content="""Build image processing request from classification and metadata.
    Input: classified email data or structured data.
    Output: JSON request to process email images.
    After building request, delegate to ExtractImageAgent using transfer_to_extract_images."""
            ),
            model_client=model_client,
            tools=[email_image_request],
            delegate_tools=[transfer_to_extract_images],
            agent_topic_type=email_image_request_topic_type,
            user_topic_type=classify_via_llm_topic_type,
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type=email_image_request_topic_type, agent_type=email_image_request_agent_type.type)
    )

    # Register ExtractImageAgent
    extract_image_agent_type = await AIAgent.register(
        runtime,
        type=extract_images_topic_type,
        factory=lambda: AIAgent(
            description="Agent to perform OCR and metadata extraction from images.",
            system_message=SystemMessage(
                content="""Extract images and perform OCR.
    Input: image request JSON.
    Output: Extracted metadata and text.
    After extraction, delegate to IngestEmbeddingAgent using transfer_to_ingest_embeddings."""
            ),
            model_client=model_client,
            tools=[extract_images],
            delegate_tools=[transfer_to_ingest_embeddings],
            agent_topic_type=extract_images_topic_type,
            user_topic_type=email_image_request_topic_type,
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type=extract_images_topic_type, agent_type=extract_image_agent_type.type)
    )

    # Register IngestEmbeddingAgent
    ingest_embedding_agent_type = await AIAgent.register(
        runtime,
        type=ingest_embeddings_topic_type,
        factory=lambda: AIAgent(
            description="Agent to ingest processed content as embeddings.",
            system_message=SystemMessage(
                content="""Ingest final processed email content into a vector database.
    Input: metadata and structured content.
    Output: Confirmation of ingestion.
    After ingestion, send result to OutputAgent."""
            ),
            model_client=model_client,
            tools=[ingest_embeddings],
            delegate_tools=[],
            agent_topic_type=ingest_embeddings_topic_type,
            user_topic_type=output_agent_topic_type,
        ),
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type=ingest_embeddings_topic_type, agent_type=ingest_embedding_agent_type.type)
    )

    # Start Pipeline
    runtime.start()

    msg = {
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

    session_id = str(uuid.uuid4())

    await runtime.publish_message(
        UserTask(context=[UserMessage(content=json.dumps(msg), source="User")]),
        topic_id=TopicId(pipeline_executor_agent_topic_type, source=session_id)
    )

    await runtime.stop_when_idle()
    await model_client.close()

# ---------------- MAIN ----------------
async def main():
    await do_create_agent()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())