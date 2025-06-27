from autogen_agentchat.agents import AssistantAgent


from autogen_core.tools import FunctionTool

from backend.app.core.email_processing_pipeline import EmailProcessingPipeline
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

from backend.config.dev_config import AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, \
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME

model_client = AzureOpenAIChatCompletionClient(
            model=AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_ad_token=AZURE_OPENAI_API_KEY,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME
        )
from autogen_ext.models.openai import OpenAIChatCompletionClient




email_pipeline = EmailProcessingPipeline()


convert_to_pdf_tool = FunctionTool(
    func=email_pipeline.convert_email_data_to_pdf,
    name="convert_email_data_to_pdf",
    description=(
        "Convert raw email data into a PDF document. "
        "Input: 'email_data' (dict) containing fields like sender, subject, body, attachments, etc. "
        """Output: A dict with {
            "path": pdf_path,
            "file_name": file_name
        }"""
    )
)

encode_file_tool = FunctionTool(
    func=email_pipeline.do_encode_via_path,
    name="do_encode_via_path",
    description=(
        "Encode a file into base64 format for safe transmission or storage. "
        "Input: 'path' (str, path) and 'file_name' (str). "
        "Output: Encoded base64 string of the file."
    )
)

paper_itemizer_tool = FunctionTool(
    func=email_pipeline.do_paper_itemizer,
    name="do_paper_itemizer",
    description=(
        "Split a PDF document into multiple image chunks (usually per page). "
        "Input: 'file_path' (str) of the PDF document. "
        "Output: List of image file paths generated from the PDF."
    )
)

classify_tool = FunctionTool(
    func=email_pipeline.do_classify_via_vlm,
    name="do_classify_via_vlm",
    description=(
        "Classify the content of the document using a Vision-Language Model (VLM). "
        "Input: 'file_path' (str) of the image or document. "
        "Output: Classification label or categories relevant to the document type."
    )
)

build_request_tool = FunctionTool(
    func=email_pipeline.build_email_image_request,
    name="build_email_image_request",
    description=(
        "Build a structured request payload for the metadata extraction step. "
        "Input: List of image file paths. "
        "Output: A formatted request dictionary containing metadata like filenames, paths, and processing context."
    )
)

upload_image_tool = FunctionTool(
    func=email_pipeline.upload_email_images,
    name="upload_email_images",
    description=(
        "Upload the document images to the storage service and extract associated metadata. "
        "Input: Prepared request object with image paths and context. "
        "Output: Metadata including image URLs, document structure, or extracted key fields."
    )
)

ingest_embedding_tool = FunctionTool(
    func=email_pipeline.ingest_all_embeddings,
    name="ingest_all_embeddings",
    description=(
        "Generate vector embeddings from the extracted metadata and document content "
        "and ingest them into the vector database for semantic search. "
        "Input: Metadata object and text content. "
        "Output: Confirmation of successful ingestion."
    )
)




# Converts raw email into PDF
doc_to_pdf_agent = AssistantAgent(
    name="DocumentToPDFAgent",
    model_client=model_client,
    tools=[convert_to_pdf_tool],
    system_message="""
You are responsible for converting email data into a PDF file.
Input: {"email_data": {...}}
Output: A dict with {
            "path": pdf_path,
            "file_name": file_name
        }
Pass this output forward for the next agent to process.
"""
)
# Encodes the PDF into base64 format
file_encoder_agent = AssistantAgent(
    name="FileEncoderAgent",
    model_client=model_client,
    tools=[encode_file_tool],
    system_message=(
        "You are responsible for encoding PDF files into base64 strings. "
        "Always use the 'do_encode_via_path' tool. "
        "Your output should include the encoded base64 content."
    )
)

# Breaks PDF into image pages
paper_itemizer_agent = AssistantAgent(
    name="PaperItemizerAgent",
    model_client=model_client,
    tools=[paper_itemizer_tool],
    system_message=(
        "You split PDF files into individual image pages for downstream processing. "
        "Use 'do_paper_itemizer' tool to perform this. "
        "Return the list of generated image file paths."
    )
)

# Classifies document types and prepares metadata requests
classifier_agent = AssistantAgent(
    name="ClassifierAgent",
    model_client=model_client,
    tools=[classify_tool, build_request_tool],
    system_message=(
        "You are responsible for two tasks:\n"
        "1. Classifying the document using the 'do_classify_via_vlm' tool.\n"
        "2. Preparing the request payload for metadata extraction with 'build_email_image_request'.\n"
        "Execute both steps in sequence if needed."
    )
)

# Uploads image pages and extracts metadata
metadata_extractor_agent = AssistantAgent(
    name="MetadataExtractorAgent",
    model_client=model_client,
    tools=[upload_image_tool],
    system_message=(
        "You handle uploading images generated from the document to storage "
        "and extract associated metadata. Use 'upload_email_images' tool. "
        "Return the extracted metadata including image URLs and document structure."
    )
)

# Ingests embeddings into vector DB
embedding_agent = AssistantAgent(
    name="EmbeddingAgent",
    model_client=model_client,
    tools=[ingest_embedding_tool],
    system_message=(
        "You are responsible for generating vector embeddings from document metadata "
        "and content, then ingesting them into the vector database for semantic search. "
        "Use 'ingest_all_embeddings' tool to perform this task."
    )
)


