from backend.src.core.email_classifier.agent_initializer import AgentInitialization


class Validator:
    def __init__(self,agent_initiate:AgentInitialization):
        self.agent_initiate = agent_initiate

    async def validate_classification(self, subject: str, body: str, classification: str) -> str:
        task = f"Email Subject: {subject}\nEmail Body: {body}\nClassification Given: {classification}"
        return await self.agent_initiate.run_agent_task(self.agent_initiate.validator_agent, task, fallback_result="Invalid")

