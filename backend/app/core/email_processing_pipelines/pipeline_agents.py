from autogen_core import TypeSubscription
from autogen_core.models import SystemMessage
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent

from backend.app.core.email_processing_pipelines.pipeline_tools import *
from backend.config.dev_config import *

model_client = AzureOpenAIChatCompletionClient(
            model=AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_ad_token=AZURE_OPENAI_API_KEY,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        )
# Delegate tools:
ttpt = FunctionTool(lambda: "DocumentToPDFAgent", description="Converts email data to PDF format.")
ttet = FunctionTool(lambda: "FileEncoderAgent", description="Encodes file at specified path to base64.")
tipt = FunctionTool(lambda: "PaperItemizerAgent", description="Processes and extracts content from paper.")
tcvt = FunctionTool(lambda: "ClassifierAgent", description="Classifies email content using VLM.")
tbrt = FunctionTool(lambda: "MetadataRequestBuilderAgent", description="Builds image request for email.")
tupt = FunctionTool(lambda: "MetadataUploaderAgent", description="Uploads images extracted from email.")
tebt = FunctionTool(lambda: "EmbeddingAgent", description="Ingests embeddings from email content.")

# Real tool wrappers:
t1 = FunctionTool(convert_email_data_to_pdf, description="Converts email data to PDF format.")
t2 = FunctionTool(do_encode_via_path, description="Encodes file at specified path to base64.")
t3 = FunctionTool(do_paper_itemizer, description="Processes and extracts content from paper.")
t4 = FunctionTool(do_classify_via_vlm, description="Classifies email content using VLM.")
t5 = FunctionTool(build_email_image_request, description="Builds image request for email.")
t6 = FunctionTool(upload_email_images, description="Uploads images extracted from email.")
t7 = FunctionTool(ingest_all_embeddings, description="Ingests embeddings from email content.")


async def register_agent(runtime, name, tool, delegates):
    await AssistantAgent.register(
        runtime, type=name,
        factory=lambda: AssistantAgent(
            description=f"{name} agent",
            system_message=SystemMessage(content=f"{name} handles one pipeline step."),
            model_client=model_client,
            tools=[tool],
            delegate_tools=delegates,
            agent_topic_type=name,
            user_topic_type="User"
        )
    )

    await runtime.add_subscription(TypeSubscription(topic_type=name, agent_type=name))

