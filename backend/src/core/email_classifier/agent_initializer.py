from backend.src.config.dev_config import *
from backend.src.core.base_agents.base_agent import BaseAgent
from backend.src.core.base_client.base_client import OpenAiClient
from backend.src.prompts.confidence_prompt import CONFIDENCE_PROMPT
from backend.src.prompts.emailclassifer_prompt import CLASSIFICATION_PROMPT, CLASSIFICATION_PROMPT_VLM
from backend.src.prompts.trigger_reason_prompt import TRIGGER_REASON_PROMPT
from backend.src.prompts.validator_prompt import VALIDATION_PROMPT
import logging

logger = logging.getLogger(__name__)

class AgentInitialization:
    def __init__(self):
        super().__init__()
        self.client = OpenAiClient()
        self.base_agent = BaseAgent(self.client.open_ai_chat_completion_client)
        self._initialize_agents()

    def _initialize_agents(self):
        self.email_classifier_agent = self.base_agent.create_assistant_agent(
            name=EMAIL_CLASSIFIER_AGENT_NAME,
            prompt=CLASSIFICATION_PROMPT
        )
        self.validator_agent = self.base_agent.create_assistant_agent(
            name=VALIDATOR_AGENT_NAME,
            prompt=VALIDATION_PROMPT
        )
        self.confidence_agent = self.base_agent.create_assistant_agent(
            name=CONFIDENCE_AGENT_NAME,
            prompt=CONFIDENCE_PROMPT
        )
        self.trigger_reason_agent = self.base_agent.create_assistant_agent(
            name=TIGGER_REASON_AGENT_NAME,
            prompt=TRIGGER_REASON_PROMPT
        )
        self.classification_with_llm = self.base_agent.create_assistant_agent(
            name=VLM_CLASSIFICATION_AGENT_NAME,
            prompt=CLASSIFICATION_PROMPT_VLM
        )
        self.summarization_agent = self.base_agent.create_assistant_agent(
            name="SUMMARIZATION_AGENT",
            prompt="You are a helpful assistant that summarizes text"
        )


    async def run_agent_task(self,agent, task: str, fallback_result=None, result_type=str):
        try:
            result = await agent.run(task=task)
            response_content = list(result.messages)[-1].content
            summary = await self.summarization_agent.run(task=response_content)
            summary_content =list(summary.messages)[-1].content
            return result_type({
                "response": response_content,
                "summary": summary_content
            })
        except Exception as e:
            logger.exception(f"❌ Agent task failed: {agent.name}")
            return fallback_result
