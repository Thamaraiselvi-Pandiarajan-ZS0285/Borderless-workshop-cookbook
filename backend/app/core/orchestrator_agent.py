from enum import Enum
import logging
from typing import Any, List, Dict, Optional
from autogen import register_function
import autogen
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm.session import sessionmaker

from backend.app.core.tools.email_parsing_agent import EmailClassificationTool
from backend.app.core.tools.retrieval_agent import RetrievalTool, retrieval_tool_fn
from backend.app.request_handler.email_request import *
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
    def __init__(self, conversation_id: str = None, db_engine:Engine = None, db_session:sessionmaker =None):
        self.conversation_id = conversation_id or f"conversation_{hash(str(id(self)))}"
        self.model_name = AZURE_OPENAI_DEPLOYMENT_NAME
        self.max_input_tokens = MAX_INPUT_TOKEN
        self.llm_config = LlmConfig().llm_config
        self.db_engine = db_engine
        self.db_session = db_session

        # self.workflow_state = WorkflowStage()
        # self.handoff_manager = HandoffManager()

        # Current workflow stage
        self.current_stage = WorkflowStage.CLASSIFICATION

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

        self.email_classification_tool = EmailClassificationTool()
        def test_email_tool(email_input: EmailClassifyImageRequest):
            self.email_classification_tool(email_input)
            return self.email_classification_tool.test

        register_function(
            test_email_tool,
            name="email_classification_tool",
            caller=self.email_classification_agent,
            executor=self.user_proxy,
            description="classifies the mail"
        )

        register_function(
            retrieval_tool_fn,
            name="retrieval_tool",
            caller=self.retrieval_agent,
            executor=self.user_proxy,
            description="Retrieves the answers from database"
        )



    def _setup_group_chat(self):
        # all the agents can be initialized here and needs to added the list below
        self.agents = [self.email_classification_agent,
                       self.orchestrator_agent,
                       self.retrieval_agent,
                       self.user_proxy]

        self.group_chat = GroupChat(
            agents=self.agents,
            messages=[],  # Shared message history
            max_round=20,
            speaker_selection_method="auto",
            allow_repeat_speaker=False,
            speaker_transitions_type="allowed"
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
            self.orchestrator_agent.initiate_chat(
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







