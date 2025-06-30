import json
import logging

from backend.src.core.email_classifier.agent_initializer import AgentInitialization

logger = logging.getLogger(__name__)


class CalculateConfidence:
    def __init__(self, agent_initiate:AgentInitialization):
        self.agent_initiate = agent_initiate

    async def calculate_confidence(self, subject: str, body: str, classification: str) -> float:
        task = f"Email Subject: {subject}\nEmail Body: {body}\nClassification Given: {classification}"
        try:
            result = await self.agent_initiate.run_agent_task(self.agent_initiate.con,task,)
            content = list(result.messages)[-1].content
            parsed = json.loads(content)
            return float(parsed.get("confidence", 0.0))
        except Exception:
            logger.exception("❌ Failed to calculate confidence.")
            return 0.0