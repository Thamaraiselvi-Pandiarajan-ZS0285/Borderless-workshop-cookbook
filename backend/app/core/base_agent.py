import asyncio
from abc import ABC, abstractmethod
from typing import Optional, List
from openai import AzureOpenAI
from backend.app.core.multi_agent_buffer import MultiAgentBuffer
from backend.config.llm_config import LlmConfig
from backend.config.dev_config import *
from openai.types.chat import ChatCompletionUserMessageParam
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import sessionmaker


class ConcreteMultiAgentBuffer(MultiAgentBuffer):
    async def on_message(self, message: str, sender: str):
        pass


class BaseAgent(ABC):
    def __init__(self, name: str, buffer: "MultiAgentBuffer", role: str = "agent"):
        self.name = name
        self.buffer = buffer
        self.role = role  # Role like 'user', 'assistant', 'system'

    @abstractmethod
    async def on_receive(self, message: str, sender: str) -> str:
        """Process incoming message and return response"""
        pass

    async def send_message(self, message: str, recipient: "BaseAgent"):
        """Send message to another agent"""
        return await recipient.on_receive(message, self.name)


class UserProxyAgent(BaseAgent):
    def __init__(self, name: str, buffer: "MultiAgentBuffer"):
        super().__init__(name, buffer, role="user")

    async def on_receive(self, message: str, sender: str) -> str:
        print(f"[{self.name}] Received message from {sender}: {message}")
        await self.buffer.persist_message(role=self.role, content=message)
        return message


class AssistantAgent(BaseAgent):
    def __init__(self, name: str, buffer: "MultiAgentBuffer",llm_config=None):
        super().__init__(name, buffer, role="assistant")
        self.llm_config = buffer.llm_config
        self.client = self._create_client()

    def _create_client(self):
        return AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
            api_version=AZURE_OPENAI_API_VERSION,
            api_key=AZURE_OPENAI_API_KEY,
        )

    async def on_receive(self, message: str, sender: str) -> str:
        print(f"[{self.name}] Processing message from {sender}: {message}")
        from openai.types.chat import ChatCompletionUserMessageParam

        messages: List[ChatCompletionUserMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content=message)
        ]
        completion = (self.client.chat.completions.create(model=AZURE_OPENAI_DEPLOYMENT_NAME,messages=messages, temperature=0.2))
        response = completion.choices[0].message.content
        await self.buffer.persist_message(role=self.role, content=response)
        return response


class GroupChatManager:
    def __init__(self, buffer: MultiAgentBuffer):
        super().__init__()
        self.buffer = buffer

    async def initiate_chat(self, initiator: str, recipient: str, message: str):
        print(f"Initiating chat from {initiator} to {recipient}: {message}")
        response = await self.buffer.route_message(message, initiator, recipient)
        print(f"Final response: {response}")
        return response

