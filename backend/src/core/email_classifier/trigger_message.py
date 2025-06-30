import logging

from backend.src.core.email_classifier.agent_initializer import AgentInitialization

logger = logging.getLogger(__name__)


class TriggerMessage:
    def __init__(self, agent_initiate:AgentInitialization):
        self.agent_initiate = agent_initiate

    async def get_trigger_message(self, subject: str, body: str, classification: str, confidence: float, validation: str = "Invalid") -> str:
        task = f"""Subject: {subject}
                    Body: {body}
                    Classification: {classification}
                    Confidence: {confidence}
                    Validation Result: {validation}"""
        return await self.agent_initiate.run_agent_task(self.agent_initiate.trigger_reason_agent, task,
                                                        fallback_result="Invalid")