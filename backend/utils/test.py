import json

from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import UserProxyAgent


from autogen_agentchat.agents import AssistantAgent

from backend.utils.orch import (
    doc_to_pdf_agent,
    file_encoder_agent,
    paper_itemizer_agent,
    embedding_agent,
    metadata_extractor_agent,
    classifier_agent,
)
from backend.config.dev_config import (
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT_NAME,
)
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient


# 🔗 Model Client Setup
model_client = AzureOpenAIChatCompletionClient(
    model=AZURE_OPENAI_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_ad_token=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
)

orchestrator_react_prompt = """
You are an orchestration agent responsible for processing emails into a pipeline.
You can think step-by-step and use other agents as tools to complete the tasks.

Follow this reasoning format strictly:

Question: {input}

Thought: Reflect on what needs to be done next in the pipeline.
Action: The next agent to delegate to. Choose one from:
[`DocumentToPDFAgent`, `FileEncoderAgent`, `PaperItemizerAgent`, `ClassifierAgent`, `MetadataExtractorAgent`, `EmbeddingAgent`, `user`]
Action Input: Information needed for the agent to perform its task.
Observation: The output or result from the agent.

(Repeat Thought → Action → Action Input → Observation as needed.)

When the process is complete:

Thought: The pipeline is complete.
Final Answer: Notify `user` and then terminate.

## Rules:
- Follow this strict pipeline order:
1. DocumentToPDFAgent
2. FileEncoderAgent
3. PaperItemizerAgent
4. ClassifierAgent
5. MetadataExtractorAgent
6. EmbeddingAgent
7. user

- Only select the next agent in this order.
- Never skip steps or loop back.

---

Begin!
"""

# 🚦 Orchestrator Agent
orchestrator_agent = AssistantAgent(
    name="OrchestratorAgent",
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    system_message=orchestrator_react_prompt,
    model_client=model_client,
)

text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=5)
termination = text_mention_termination | max_messages_termination

selector_prompt= selector_prompt = """
You are a selector agent responsible for managing an email processing pipeline.

There are the following agents in the pipeline, and they must act in this strict order:

1. DocumentToPDFAgent
2. FileEncoderAgent
3. PaperItemizerAgent
4. ClassifierAgent
5. MetadataExtractorAgent
6. EmbeddingAgent
7. user

## Rules:
- Always follow this order strictly without skipping or rearranging.
- Only one agent should act at a time.
- Once EmbeddingAgent finishes, return control to `user` and then terminate the conversation.
- Do not select an agent who has already finished their task unless it's `user` at the end.

## Output Format:
- Only respond with the name of the next agent from this list:
[`DocumentToPDFAgent`, `FileEncoderAgent`, `PaperItemizerAgent`, `ClassifierAgent`, `MetadataExtractorAgent`, `EmbeddingAgent`, `user`]

- Do NOT include any reasoning, explanation, or extra text — only the agent name.

## If uncertain:
- Default to the next agent in the pipeline order based on the last completed agent.

---
"""


def react_prompt_message(sender, recipient, context):
    return selector_prompt.format(input=context["question"])

user = UserProxyAgent(name="user")

team = SelectorGroupChat(
    [
        orchestrator_agent,
        doc_to_pdf_agent,
        file_encoder_agent,
        paper_itemizer_agent,
        classifier_agent,
        metadata_extractor_agent,
        embedding_agent,
        user,  # ✅ User added here
    ],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,
)

# Input Email
user_message = {
    "sender": "abc@example.com",
    "received_at": "2025-06-19T09:40:11.106000+00:00",
    "subject": "Bid Submission",
    "body": "Please find attached the bid...",
    "hasAttachments": False,
    "attachments": [],
}

# Async Pipeline Run
import asyncio
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console

async def run_pipeline(user_messages):
    text_message = TextMessage(
        content=json.dumps({"email_data": user_messages}),
        source="user"  # ✅ Fixed to match the agent name
    )
    await Console(team.run_stream(task=text_message))

asyncio.run(run_pipeline(user_message))