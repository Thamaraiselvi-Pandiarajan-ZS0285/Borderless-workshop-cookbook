from enum import Enum
import logging
from typing import Any
from autogen import register_function
import autogen
from backend.app.request_handler.email_request import EmailClassificationRequest
from backend.prompts.orchestration_prompt import *
from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from backend.config.dev_config import *
from backend.config.llm_config import LlmConfig
from autogen import AssistantAgent
from backend.app.core.classifier_agent import EmailClassifierProcessor
from backend.app.request_handler.orchestrator_protocol import WorkflowStage
from backend.app.core.hand_off_manager import HandoffManager
from backend.app.request_handler.orchestrator_protocol import OrchestrateRequest


class Orchestrator():
    def __init__(self):
        self.model_name = AZURE_OPENAI_DEPLOYMENT_NAME
        self.max_input_tokens = MAX_INPUT_TOKEN
        self.llm_config = LlmConfig().llm_config

        # self.workflow_state = WorkflowStage()
        # self.handoff_manager = HandoffManager()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.user_proxy = autogen.UserProxyAgent(
                        name="UserProxy",
                        human_input_mode="NEVER",  # Only needed for initial message
                        code_execution_config=False,
                        is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
                    )
        self.orchestrator_agent = autogen.UserProxyAgent(
                name=ORCHESTRATOR_AGENT_NAME,
                system_message=ORCHESTRATOR_PROMPT,
                human_input_mode="NEVER",  # Only needed for initial message
                code_execution_config=False,
                is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
        )
        # self.orchestrator_agent = self._create_agent(ORCHESTRATOR_AGENT_NAME, ORCHESTRATOR_PROMPT,
        #                                              "Main orchestrator that coordinates all workflow stages")
        self.email_classification_agent = self._create_agent(EMAIL_CLASSIFICATION_AGENT_NAME, EMAIL_CLASSIFICATION_PROMPT,
                                                     "Classify the email")


    def _create_agent(self, name: str, prompt: str, description: str = None) -> AssistantAgent:
        return AssistantAgent(
            name=name,
            system_message=prompt,
            llm_config=self.llm_config,
            description=description
        )

    def _initialize_agents(self):
        # self.email_classification_tool = EmailClassifierProcessor()

        def email_classification_tool(subject: str, body: str) -> dict:
            return EmailClassifierProcessor.process_email(subject, body)
        register_function(
            email_classification_tool,
            name="email_classification_tool",
            caller=self.email_classification_agent,
            executor=self.user_proxy,
            description="classifies the mail"
        )
        #all the agents can be initialized here and needs to added the list below
        self.agents = [self.email_classification_agent, self.orchestrator_agent, self.user_proxy]


    def orchestrate(self, message: str) -> str:
        return self.orchestrator_agent.generate_reply(
            messages=[{"role": "user", "content": message}]
        ).strip()

    def create_group_chat(self) -> GroupChat:
        """Create group chat with all agents"""

        group_chat = GroupChat(
            agents=self.agents,
            messages=[],
            max_round=50,
            speaker_selection_method="auto",
            allow_repeat_speaker=False
        )

        return group_chat

    def create_group_chat_manager(self, group_chat: GroupChat) -> GroupChatManager:
        """Create group chat manager"""

        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=self.llm_config,
            system_message="""
            You are managing a market research workflow with multiple specialized agents starting with orchestrator agent
            Ensure proper handoffs between agents and maintain workflow state.
            Coordinate human-in-the-loop interactions when needed.
            """
        )

        return manager
