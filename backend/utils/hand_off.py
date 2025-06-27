import logging
import os
import json
import uuid
from typing import List, Tuple

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

from backend.app.core.email_to_pdf_converter import HTMLEmailToPDFConverter
from backend.app.core.file_operations import FileToBase64
from backend.app.request_handler.paper_itemizer import PaperItemizerRequest
from backend.config.dev_config import DEFAULT_IMAGE_FORMAT

# ----------------------- Configure Logging -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EmailProcessingPipeline")


AZURE_OPENAI_API_KEY="2K3oZXs28WZE2y1Fzg1jIPUdPGSY3xF0cWPcDx2DlF4RpUKimG0DJQQJ99BFACYeBjFXJ3w3AAABACOG9Osf"
AZURE_OPENAI_ENDPOINT= "https://bap-openai.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME="gpt-41-mini"
AZURE_OPENAI_API_VERSION="2025-01-01-preview"
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1-mini"


config_list = [
    {
        "model": "gpt-4.1-mini",
        "api_type": "azure",
        "api_key": AZURE_OPENAI_API_KEY,
        "base_url":AZURE_OPENAI_ENDPOINT,
        "api_version":AZURE_OPENAI_API_VERSION,
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_email_data_to_pdf(email_data: dict) -> dict:
    if not isinstance(email_data, dict):
        raise ValueError("The input must be a dictionary.")

    output_dir = "/home/dinesh.krishna@zucisystems.com/workspace/data/"
    os.makedirs(output_dir, exist_ok=True)
    file_name = str(uuid.uuid4())
    pdf_path = os.path.join(output_dir, f"{file_name}.pdf")

    pdf_converter = HTMLEmailToPDFConverter()
    pdf_converter.convert_to_pdf(email_data, pdf_path)

    logger.info(f"✅ PDF generated at: {pdf_path}")
    return {
        "path": pdf_path,
        "file_name": file_name
    }
def do_encode_via_path(path:str, file_name:str):

    # Double-check path exists
    if not os.path.isfile(path):
        raise

    # Perform encoding
    try:
        base64_encoder = FileToBase64(str(path))
        encoded_data = base64_encoder.do_base64_encoding_by_file_path()
    except Exception as e:
        raise e
    paper_itemizer_object = PaperItemizerRequest(
        input=encoded_data,
        file_name=file_name,
        file_extension=DEFAULT_IMAGE_FORMAT)
    return paper_itemizer_object

class UserLogin(BaseModel):
    pass


class UserTask(BaseModel):
    context: List[LLMMessage]


class AgentResponse(BaseModel):
    reply_to_topic_type: str
    context: List[LLMMessage]





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
        # Send the task to the LLM.
        llm_result = await self._model_client.create(
            messages=[self._system_message] + message.context,
            tools=self._tool_schema + self._delegate_tool_schema,
            cancellation_token=ctx.cancellation_token,
        )
        print(f"{'-'*80}\n{self.id.type}:\n{llm_result.content}", flush=True)
        # Process the LLM result.
        while isinstance(llm_result.content, list) and all(isinstance(m, FunctionCall) for m in llm_result.content):
            tool_call_results: List[FunctionExecutionResult] = []
            delegate_targets: List[Tuple[str, UserTask]] = []
            # Process each function call.
            for call in llm_result.content:
                arguments = json.loads(call.arguments)
                if call.name in self._tools:
                    # Execute the tool directly.
                    result = await self._tools[call.name].run_json(arguments, ctx.cancellation_token)
                    result_as_str = self._tools[call.name].return_value_as_string(result)
                    tool_call_results.append(
                        FunctionExecutionResult(call_id=call.id, content=result_as_str, is_error=False, name=call.name)
                    )
                elif call.name in self._delegate_tools:
                    # Execute the tool to get the delegate agent's topic type.
                    result = await self._delegate_tools[call.name].run_json(arguments, ctx.cancellation_token)
                    topic_type = self._delegate_tools[call.name].return_value_as_string(result)
                    # Create the context for the delegate agent, including the function call and the result.
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
                    raise ValueError(f"Unknown tool: {call.name}")
            if len(delegate_targets) > 0:
                # Delegate the task to other agents by publishing messages to the corresponding topics.
                for topic_type, task in delegate_targets:
                    print(f"{'-'*80}\n{self.id.type}:\nDelegating to {topic_type}", flush=True)
                    await self.publish_message(task, topic_id=TopicId(topic_type, source=self.id.key))
            if len(tool_call_results) > 0:
                print(f"{'-'*80}\n{self.id.type}:\n{tool_call_results}", flush=True)
                # Make another LLM call with the results.
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
                print(f"{'-'*80}\n{self.id.type}:\n{llm_result.content}", flush=True)
            else:
                # The task has been delegated, so we are done.
                return
        # The task has been completed, publish the final result.
        assert isinstance(llm_result.content, str)
        message.context.append(AssistantMessage(content=llm_result.content, source=self.id.type))
        await self.publish_message(
            AgentResponse(context=message.context, reply_to_topic_type=self._agent_topic_type),
            topic_id=TopicId(self._user_topic_type, source=self.id.key),
        )


class HumanAgent(RoutedAgent):
    def __init__(self, description: str, agent_topic_type: str, user_topic_type: str) -> None:
        super().__init__(description)
        self._agent_topic_type = agent_topic_type
        self._user_topic_type = user_topic_type

    @message_handler
    async def handle_user_task(self, message: UserTask, ctx: MessageContext) -> None:
        human_input = input("Human agent input: ")
        print(f"{'-'*80}\n{self.id.type}:\n{human_input}", flush=True)
        message.context.append(AssistantMessage(content=human_input, source=self.id.type))
        await self.publish_message(
            AgentResponse(context=message.context, reply_to_topic_type=self._agent_topic_type),
            topic_id=TopicId(self._user_topic_type, source=self.id.key),
        )


class UserAgent(RoutedAgent):
    def __init__(self, description: str, user_topic_type: str, agent_topic_type: str) -> None:
        super().__init__(description)
        self._user_topic_type = user_topic_type
        self._agent_topic_type = agent_topic_type

    @message_handler
    async def handle_user_login(self, message: UserLogin, ctx: MessageContext) -> None:
        print(f"{'-' * 80}\nUser login, session ID: {self.id.key}.", flush=True)
        # Get the user's initial input after login.
        user_input = input("User: ")
        print(f"{'-' * 80}\n{self.id.type}:\n{user_input}")
        await self.publish_message(
            UserTask(context=[UserMessage(content=user_input, source="User")]),
            topic_id=TopicId(self._agent_topic_type, source=self.id.key),
        )

    @message_handler
    async def handle_task_result(self, message: AgentResponse, ctx: MessageContext) -> None:
        # Get the user's input after receiving a response from an agent.
        user_input = input("User (type 'exit' to close the session): ")
        print(f"{'-'*80}\n{self.id.type}:\n{user_input}", flush=True)
        if user_input.strip().lower() == "exit":
            print(f"{'-'*80}\nUser session ended, session ID: {self.id.key}.")
            return
        message.context.append(UserMessage(content=user_input, source="User"))
        await self.publish_message(
            UserTask(context=message.context), topic_id=TopicId(message.reply_to_topic_type, source=self.id.key)
        )

convert_email_into_pdf = FunctionTool(convert_email_data_to_pdf, description="Convert the email data into pdf")
encode_pdf = FunctionTool(
    do_encode_via_path, description="Encode the pdf file"
)


convert_email_agent_name = "ConvertEmailToPdf"
encode_agent_topic_type = "EncodePdf"
triage_agent_topic_type = "TriageAgent"
human_agent_topic_type = "HumanAgent"
user_topic_type_ = "User"


def transfer_to_email_to_pdf() -> str:
    return convert_email_agent_name


def transfer_to_encode_agent() -> str:
    return encode_agent_topic_type


def transfer_back_to_triage() -> str:
    return triage_agent_topic_type


def escalate_to_human() -> str:
    return human_agent_topic_type


transfer_to_pdf_converstion_agent_tools = FunctionTool(
    transfer_to_email_to_pdf, description="Use for email to pdf conversion"
)
transfer_to_encode_agent_tool = FunctionTool(
    transfer_to_encode_agent, description="Use for encode files"
)
transfer_back_to_triage_tool = FunctionTool(
    transfer_back_to_triage,
    description="Call this if the user brings up a topic outside of your purview,\nincluding escalating to human.",
)
escalate_to_human_tool = FunctionTool(escalate_to_human, description="Only call this if explicitly asked to.")


async def do_create_agent():

    runtime = SingleThreadedAgentRuntime()

    # Register the triage agent.
    triage_agent_type = await AIAgent.register(
        runtime,
        type=triage_agent_topic_type,  # Using the topic type as the agent type.
        factory=lambda: AIAgent(
            description="A triage agent.",
            system_message=SystemMessage(
                content="""You are a selector agent responsible for managing an email processing pipeline.
    
    There are the following agents in the pipeline, and they must act in this strict order:
    
    1. DocumentToPDFAgent
    2. FileEncoderAgent
    
    
    ## Rules:
    - Always follow this order strictly without skipping or rearranging.
    - Only one agent should act at a time.
    - Once EmbeddingAgent finishes, return control to `user` and then terminate the conversation.
    - Do not select an agent who has already finished their task unless it's `user` at the end.
    
    ## Output Format:
    - Only respond with the name of the next agent from this list:
    [`DocumentToPDFAgent`, `FileEncoderAgent`]
    
    - Do NOT include any reasoning, explanation, or extra text — only the agent name.
    
    ## If uncertain:
    - Default to the next agent in the pipeline order based on the last completed agent.
    
    ---"""
            ),
            model_client=model_client,
            tools=[],
            delegate_tools=[
                transfer_to_pdf_converstion_agent_tools,
                transfer_to_encode_agent_tool,
                escalate_to_human_tool
            ],
            agent_topic_type=triage_agent_topic_type,
            user_topic_type=user_topic_type_,
        ),
    )
    await runtime.add_subscription(TypeSubscription(topic_type=triage_agent_topic_type, agent_type=triage_agent_type.type))

    # Register the sales agent.
    convert_email_agent_type = await AIAgent.register(
        runtime,
        type=convert_email_agent_name,  # Using the topic type as the agent type.
        factory=lambda: AIAgent(
            description="email to pdf agent.",
            system_message=SystemMessage(
                content="Convert raw email data into a PDF document. "
                        "Input: 'email_data' (dict) containing fields like sender, subject, body, attachments, etc. "
                        """Output: A dict with {
                             "path": pdf_path,
                             "file_name": file_name
                         }"""
            ),
            model_client=model_client,
            tools=[convert_email_data_to_pdf],
            delegate_tools=[transfer_back_to_triage_tool],
            agent_topic_type=convert_email_agent_name,
            user_topic_type=user_topic_type_,
        ),
    )
    # Add subscriptions for the sales agent: it will receive messages published to its own topic only.
    await runtime.add_subscription(
        TypeSubscription(topic_type=convert_email_agent_name, agent_type=convert_email_agent_type.type))

    # Register the issues and repairs agent.
    encode_agent_type = await AIAgent.register(
        runtime,
        type=encode_agent_topic_type,  # Using the topic type as the agent type.
        factory=lambda: AIAgent(
            description="An Encoder agent.",
            system_message=SystemMessage(
                content="""Encode a file into base64 format for safe transmission or storage. "
            "Input: 'path' (str, path) and 'file_name' (str). "
            "Output: Encoded base64 string of the file."""
            ),
            model_client=model_client,
            tools=[
                encode_pdf,
            ],
            delegate_tools=[transfer_to_encode_agent_tool],
            agent_topic_type=encode_agent_topic_type,
            user_topic_type=user_topic_type_,
        ),
    )
    # Add subscriptions for the issues and repairs agent: it will receive messages published to its own topic only.
    await runtime.add_subscription(
        TypeSubscription(topic_type=encode_agent_topic_type, agent_type=encode_agent_type.type)
    )

    # Register the human agent.
    human_agent_type = await HumanAgent.register(
        runtime,
        type=human_agent_topic_type,  # Using the topic type as the agent type.
        factory=lambda: HumanAgent(
            description="A human agent.",
            agent_topic_type=human_agent_topic_type,
            user_topic_type=user_topic_type_,
        ),
    )
    # Add subscriptions for the human agent: it will receive messages published to its own topic only.
    await runtime.add_subscription(TypeSubscription(topic_type=human_agent_topic_type, agent_type=human_agent_type.type))

    # Register the user agent.
    user_agent_type = await UserAgent.register(
        runtime,
        type=user_topic_type_,
        factory=lambda: UserAgent(
            description="A user agent.",
            user_topic_type=user_topic_type_,
            agent_topic_type=triage_agent_topic_type,  # Start with the triage agent.
        ),
    )
    # Add subscriptions for the user agent: it will receive messages published to its own topic only.
    await runtime.add_subscription(TypeSubscription(topic_type=user_topic_type_, agent_type=user_agent_type.type))

    msg= {
    "sender": "koumiya",
      "subject": "Q2 Financial Report Submission",
      "received_at": "2025-06-19T09:40:11.106Z",
      "subject": "string",
      "body": "Dear Akhil,We are delighted to inform you that **Borderless Access** has been selected as the awarded vendor for the **Multi-Country Healthcare Market Research** project.Your proposal demonstrated exceptional understanding of the healthcare domain and offered a technically sound and cost-effective methodology. Your localized panel recruitment strategy and clear compliance planning were highly appreciated by the evaluation committee.Our operations team will reach out within the next week to schedule a formal project kickoff on **July 16, 2025**.Looking forward to a successful partnership.Warm regards,**Radhika Mehta**Insights LeadHealthWorld Insights Consortium",
      "hasAttachments": False,
      "attachments": []
    }

    runtime.start()

    # Create a new session for the user.
    session_id = str(uuid.uuid4())
    await runtime.publish_message(UserLogin(), topic_id=TopicId(user_topic_type_, source=session_id))

    # Run until completion.
    await runtime.stop_when_idle()
    await model_client.close()


async def main():
    await do_create_agent()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
