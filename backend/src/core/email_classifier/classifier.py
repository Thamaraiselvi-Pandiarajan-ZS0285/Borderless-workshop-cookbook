from backend.src.core.email_classifier.agent_initializer import AgentInitialization
from backend.src.core.email_classifier.classifier_interface import ClassifierInterface


class Classifier(ClassifierInterface):
    def __init__(self, agent_initializer: AgentInitialization):
        super().__init__()
        self.agent_initializer = agent_initializer

    async def classify(self, input_data: str) -> str:
        return await self.agent_initializer.run_agent_task(
            self.agent_initializer.email_classifier_agent,
            task=input_data,
            fallback_result="Unclear"
        )
