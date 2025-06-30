from autogen_core import TypeSubscription
from autogen_core.models import SystemMessage
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from pipeline_tools import *
from pipeline_tools import (
    convert_email_data_to_pdf,
    do_encode_via_path,
    do_paper_itemizer,
    do_classify_via_vlm,
    build_email_image_request,
    upload_email_images,
    ingest_all_embeddings,
)

model_client = AzureOpenAIChatCompletionClient(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_ad_token=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        )
# Delegate tools:
ttpt = FunctionTool(lambda: "DocumentToPDFAgent")
ttet = FunctionTool(lambda: "FileEncoderAgent")
tipt = FunctionTool(lambda: "PaperItemizerAgent")
tcvt = FunctionTool(lambda: "ClassifierAgent")
tbrt = FunctionTool(lambda: "MetadataRequestBuilderAgent")
tupt = FunctionTool(lambda: "MetadataUploaderAgent")
tebt = FunctionTool(lambda: "EmbeddingAgent")

# Real tool wrappers:
t1=FunctionTool(convert_email_data_to_pdf); t2=FunctionTool(do_encode_via_path)
t3=FunctionTool(do_paper_itemizer); t4=FunctionTool(do_classify_via_vlm)
t5=FunctionTool(build_email_image_request); t6=FunctionTool(upload_email_images)
t7=FunctionTool(ingest_all_embeddings)

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
