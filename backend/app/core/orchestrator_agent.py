from enum import Enum
import logging
from typing import Any, List, Dict, Optional
from autogen import register_function
import autogen

from backend.app.core.tools.email_parsing_agent import EmailClassificationTool
from backend.app.request_handler.email_request import EmailClassificationRequest
from backend.prompts.orchestration_prompt import *
from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from backend.config.dev_config import *
from backend.config.llm_config import LlmConfig
from autogen import AssistantAgent
from backend.app.core.classifier_agent import EmailClassifierProcessor
# from backend.app.request_handler.orchestrator_protocol import WorkflowStage
# from backend.app.core.hand_off_manager import HandoffManager
from backend.app.request_handler.orchestrator_protocol import OrchestrateRequest
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen.agentchat import ConversableAgent
from backend.prompts.retrieval_prompt import RETRIEVAL_PROMPT

class WorkflowStage(Enum):
    CLASSIFICATION = "classification"
    RETRIEVAL = "retrieval"
    ANALYSIS = "analysis"
    COMPLETION = "completion"

class Orchestrator:
    def __init__(self, conversation_id: str = None):
        self.conversation_id = conversation_id or f"conversation_{hash(str(id(self)))}"
        self.model_name = AZURE_OPENAI_DEPLOYMENT_NAME
        self.max_input_tokens = MAX_INPUT_TOKEN
        self.llm_config = LlmConfig().llm_config

        # self.workflow_state = WorkflowStage()
        # self.handoff_manager = HandoffManager()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize agents
        self._initialize_agents()
        self._setup_tools()

        # Setup group chat with shared memory
        self._setup_group_chat()

    def _create_agent(self, name: str, prompt: str, description: str = None) -> AssistantAgent:
        return AssistantAgent(
            name=name,
            system_message=prompt,
            llm_config=self.llm_config,
            description=description,
            context_variables= UnboundedChatCompletionContext(),
            chat_messages = []
        )

    def _initialize_agents(self):
        self.user_proxy = autogen.UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",  # Only needed for initial message
            code_execution_config=False,
            is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
            chat_messages=[]
        )

        self.orchestrator_agent = autogen.UserProxyAgent(
            name=ORCHESTRATOR_AGENT_NAME,
            system_message=ORCHESTRATOR_PROMPT,
            code_execution_config=False,
            is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
            chat_messages=[]
        )

        self.email_classification_agent = self._create_agent(EMAIL_CLASSIFIER_AGENT_NAME, EMAIL_CLASSIFICATION_PROMPT,
                                                             "Classify the email")

        self.retrieval_agent = self._create_agent(RETRIEVAL_AGENT_NAME, RETRIEVAL_PROMPT,
                                                             "Use RAG to retrieve the context")



    def _setup_tools(self):
        self.email_classification_tool = EmailClassificationTool(email_file=EmailClassificationRequest()).test()

        register_function(
            self.email_classification_tool,
            name="email_classification_tool",
            caller=self.email_classification_agent,
            executor=self.user_proxy,
            description="classifies the mail"
        )

    def _setup_group_chat(self):
        # all the agents can be initialized here and needs to added the list below
        self.agents = [self.email_classification_agent,
                       self.orchestrator_agent,
                       self.user_proxy]

        self.group_chat = GroupChat(
            agents=self.agents,
            messages=[],  # Shared message history
            max_round=20,
            speaker_selection_method="manual",  # Use manual for handoff pattern
            allow_repeat_speaker=True,
            speaker_transitions_type="allowed",
            allowed_or_disallowed_speaker_transitions={
                self.orchestrator_agent: [self.email_classification_agent],
                self.email_classification_agent: [self.retrieval_agent, self.orchestrator_agent],
                self.retrieval_agent: [self.orchestrator_agent],
                self.analysis_agent: [self.orchestrator_agent],
                self.user_proxy: [self.orchestrator_agent]
            }
        )

        # Group Chat Manager with handoff logic
        self.group_chat_manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
            system_message= f"""
            You are managing an email processing workflow with conversation ID: {self.conversation_id}
            
            Agent responsibilities:
            - Orchestrator: Manages overall workflow and stage transitions
            - EmailClassifier: Classifies and categorizes emails
            - RetrievalAgent: Retrieves relevant information and context
            - AnalysisAgent: Analyzes and processes information
            
            Handoff Pattern:
            1. Start with Orchestrator
            2. Orchestrator → EmailClassifier for classification
            3. EmailClassifier → RetrievalAgent for information gathering
            4. RetrievalAgent → AnalysisAgent for processing
            5. AnalysisAgent → Orchestrator for completion
            
            Monitor for "HANDOFF_TO: [agent_name]" to manage transitions.
            Maintain conversation context and shared memory across all agents.
            """,
            is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", "").upper()
        )

    def orchestrate(self,message:str) -> str:
        """Main orchestration method using group chat"""
        try:
            self.logger.info(f"Starting orchestration for conversation: {self.conversation_id}")
            self.logger.info(f"Initial message: {message}")

            # Start the group chat conversation
            self.user_proxy.initiate_chat(
                self.group_chat_manager,
                message=message,
                max_turns=20,
                clear_history=False  # Maintain chat history
            )

            # Extract final response from chat history
            if self.group_chat.messages:
                final_message = self.group_chat.messages[-1]
                return final_message.get("content", "No response generated")

            return "Orchestration completed successfully"

        except Exception as e:
            self.logger.error(f"Error in orchestration: {str(e)}")
            return f"Error occurred during orchestration: {str(e)}"

    def _parse_handoff(self, message: str) -> Optional[str]:
        """Parse handoff instruction from message"""
        if "HANDOFF_TO:" in message.upper():
            parts = message.upper().split("HANDOFF_TO:")
            if len(parts) > 1:
                target_agent = parts[1].strip().split()[0]
                return target_agent
        return None

    def _get_agent_by_name(self, name: str) -> Optional[ConversableAgent]:
        """Get agent by name"""
        name_mapping = {
            "ORCHESTRATOR": self.orchestrator_agent,
            "EMAILCLASSIFIER": self.email_classification_agent,
            "RETRIEVALAGENT": self.retrieval_agent,
            "USERPROXY": self.user_proxy
        }
        return name_mapping.get(name.upper())
    def get_conversation_history(self) -> List[Dict]:
        """Get the full conversation history"""
        return self.group_chat.messages

    def get_agent_chat_history(self, agent_name: str) -> List[Dict]:
        """Get chat history for a specific agent"""
        agent = self._get_agent_by_name(agent_name)
        if agent and hasattr(agent, 'chat_messages'):
            print(agent.chat_messages)
            return agent.chat_messages
        return agent.chat_messages

    def clear_conversation_history(self):
        """Clear conversation history (if needed)"""
        self.group_chat.messages.clear()
        for agent in self.agents:
            if hasattr(agent, 'chat_messages'):
                agent.chat_messages.clear()

    def get_current_stage(self) -> WorkflowStage:
        """Get current workflow stage"""
        return self.current_stage

    def set_stage(self, stage: WorkflowStage):
        """Set current workflow stage"""
        self.current_stage = stage
        self.logger.info(f"Workflow stage updated to: {stage.value}")

    def resume_conversation(self, message: str = None) -> str:
        """Resume existing conversation with optional new message"""
        if message:
            return self.orchestrate(message)
        else:
            # Resume with last context
            if self.group_chat.messages:
                last_message = self.group_chat.messages[-1]
                return self.orchestrate(f"Continue from: {last_message.get('content', '')}")
            return "No previous conversation to resume"





