from typing import Union, List
from backend.src.core.email_classifier.agent_initializer import AgentInitialization
from backend.src.utils.batch_async.async_batch import AsyncBatchUtil


class Classifier:
    def __init__(self, agent_initializer: AgentInitialization):
        self.agent_initializer = agent_initializer

    async def classify(self, input: Union[str, List[str]], max_concurrency: int = 4):
        if isinstance(input, list):
            return await self._classify_batch(input, max_concurrency)
        else:
            return await self._classify_single(input)

    async def _classify_single(self, input_data: str) -> str:
        return await self.agent_initializer.run_agent_task(
            self.agent_initializer.email_classifier_agent,
            task=input_data,
            fallback_result="Unclear"
        )

    async def _classify_batch(self, emails: List[str], max_concurrency: int = 5):
        return await AsyncBatchUtil.run_batch(
            inputs=emails,
            async_func=self._classify_single,
            max_concurrency=max_concurrency
        )
