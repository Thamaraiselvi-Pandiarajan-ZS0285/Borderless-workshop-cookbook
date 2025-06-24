import asyncio
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
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.teams import SelectorGroupChat, Swarm
from autogen_agentchat.messages import TextMessage, HandoffMessage
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from backend.app.core.classifier_agent import EmailClassifierProcessor
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
        self.user_memory = ListMemory()

        # self.workflow_state = WorkflowStage()
        # self.handoff_manager = HandoffManager()

        # Current workflow stage
        self.current_stage = WorkflowStage.CLASSIFICATION
        self._initialize_model_client()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize agents
        self._initialize_agents()
        self._setup_tools()

        # Setup group chat with shared memory
        self._setup_team_chat()

    def _initialize_model_client(self):
        """Initialize the OpenAI model client"""
        self.model_client = AzureOpenAIChatCompletionClient(
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
            model=AZURE_OPENAI_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            model_capabilities={
                "vision": False,
                "function_calling": True,
                "json_output": True,
            }
        )

    def _initialize_agents(self):
        self.orchestrator_agent = AssistantAgent(
            name=ORCHESTRATOR_AGENT_NAME,
            model_client=self.model_client,
            system_message=f"""{ORCHESTRATOR_PROMPT} \n
            You are the orchestrator for conversation: {self.conversation_id}
            Your responsibilities:
            1. Manage workflow stages: {[stage.value for stage in WorkflowStage]}
            2. Coordinate handoffs between agents
            3. Monitor progress and ensure completion
            
            Use HANDOFF messages to transfer control to specific agents.
            Current stage: {self.current_stage.value}
            """,
            tools=[],
            memory=[self.user_memory]
        )

        # Email Classification Agent
        self.email_classification_agent = AssistantAgent(
            name=EMAIL_CLASSIFIER_AGENT_NAME,
            model_client=self.model_client,
            system_message=f"""
                            {EMAIL_CLASSIFICATION_PROMPT}\n
                            You are responsible for classifying emails in conversation: {self.conversation_id}
                            where you also will do the indexing and ingest the embeddings along with the data to the database
                            After classification, hand off to the retrieval agent for context gathering from db.
                            """,
            tools=self._get_classification_tools(),
            memory=[self.user_memory]
        )

        # Retrieval Agent
        self.retrieval_agent = AssistantAgent(
            name=RETRIEVAL_AGENT_NAME,
            model_client=self.model_client,
            system_message=f"""
                           {RETRIEVAL_PROMPT}
                           You are responsible for retrieving relevant information in conversation: {self.conversation_id}
                           Use RAG to find contextual information based on the classified email.
                           """,
            tools=self._get_retrieval_tools(),
            memory=[self.user_memory]
        )

    def _get_classification_tools(self):
        """Get tools for email classification agent"""

        async def email_classification_tool(email_input: Dict[str,Any]) -> str:
            """Classify the email content"""
            try:
                # Convert string input to proper request format if needed
                classification_request = EmailClassifyImageRequest(content=email_input)
                tool = EmailClassificationTool()
                result = tool(classification_request)
                return f"Email classified successfully: {result}"
            except Exception as e:
                return f"Classification error: {str(e)}"

        return [email_classification_tool]

    def _get_retrieval_tools(self):
        """Get tools for retrieval agent"""

        async def retrieval_tool(query: str) -> str:
            """Retrieve relevant information from database"""
            try:
                result = retrieval_tool_fn(query)
                return f"Retrieved information: {result}"
            except Exception as e:
                return f"Retrieval error: {str(e)}"
        return [retrieval_tool]

    # def _setup_tools(self):
    #     """Setup tools for agents - kept for compatibility"""
    #     self.email_classification_tool = EmailClassificationTool()

    def _setup_team_chat(self):
        """Setup the team chat for agent coordination"""

        # Define the agents list
        self.agents = [
            self.orchestrator_agent,
            self.email_classification_agent,
            self.retrieval_agent
        ]

        # Create termination condition
        self.termination_condition = TextMentionTermination("TERMINATE")

        # Setup RoundRobin team for structured workflow
        self.team_chat = SelectorGroupChat(
            participants=self.agents,
            termination_condition=self.termination_condition,
            max_turns=2,
            model_client=self.model_client
        )

        # Alternative: Setup Swarm for more dynamic handoffs
        # self.swarm_chat = Swarm(
        #     participants=self.agents,
        #     termination_condition=self.termination_condition
        # )

    async def orchestrate_async(self, message: str) -> str:
        """Async orchestration method using team chat"""
        try:
            self.logger.info(f"Starting async orchestration for conversation: {self.conversation_id}")
            self.logger.info(f"Initial message: {message}")

            # Create initial message
            initial_message = TextMessage(
                content=f"""
                Here is the initial user query for conversation {self.conversation_id}:

                {message}

                Please begin with email classification, then proceed with retrieval and analysis based on the input query
                """,
                source=ORCHESTRATOR_AGENT_NAME
            )

            # Run the team chat
            result_stream = self.team_chat.run_stream(task=initial_message)

            final_response = ""
            async for message in result_stream:
                if hasattr(message, 'content'):
                    final_response = message.content
                    # self.logger.info(f"Received message: {message.content[:100]}...")

            return final_response or "Orchestration completed successfully"

        except Exception as e:
            self.logger.error(f"Error in async orchestration: {str(e)}")
            return f"Error occurred during orchestration: {str(e)}"

    def orchestrate(self, message: str) -> str:
        """Synchronous wrapper for orchestration"""
        try:
            # Run the async orchestration
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.orchestrate_async(message))
            loop.close()
            return result
        except Exception as e:
            self.logger.error(f"Error in orchestration: {str(e)}")
            return f"Error occurred during orchestration: {str(e)}"

    def update_workflow_stage(self, new_stage: WorkflowStage):
        """Update the current workflow stage"""
        self.current_stage = new_stage
        self.logger.info(f"Workflow stage updated to: {new_stage.value}")

    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history from team chat"""
        try:
            # This would depend on the specific implementation of team chat history
            return []  # Placeholder - implement based on autogen_agentchat history access
        except Exception as e:
            self.logger.error(f"Error getting conversation history: {str(e)}")
            return []

    def reset_conversation(self):
        """Reset the conversation state"""
        self.current_stage = WorkflowStage.CLASSIFICATION
        # Reset team chat if needed
        self._setup_team_chat()
        self.logger.info(f"Conversation {self.conversation_id} reset")








