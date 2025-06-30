from abc import ABC, abstractmethod
from typing import Any, List

from backend.src.core.email_classifier.agent_initializer import AgentInitialization


class ClassifierInterface(ABC):
    def __init__(self):
        self.agent_initiate = AgentInitialization

    @abstractmethod
    async def classify(self, input_data: Any) -> Any:
        """Subclasses must define how to classify a single input."""
        pass

    async def classify_batch(self, batch_data: List[Any]) -> List[Any]:
        results = []
        for item in batch_data:
            result = await self.classify(item)
            results.append(result)
        return results

