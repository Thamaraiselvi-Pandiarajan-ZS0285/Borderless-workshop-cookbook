from typing import List, Callable, Awaitable, Any
from asyncio import as_completed
import asyncio

class AsyncBatchUtil:
    @staticmethod
    async def run_batch(inputs: List[Any],
        async_func: Callable[[Any], Awaitable[Any]],
        max_concurrency: int = 5
    ) -> List[Any]:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def sem_task(input_item, idx):
            async with semaphore:
                try:
                    result = await async_func(input_item)
                    return idx, result
                except Exception as e:
                    return idx, {"input": input_item, "error": str(e)}

        tasks = [sem_task(item, i) for i, item in enumerate(inputs)]
        results = [None] * len(inputs)

        for future in as_completed(tasks):
            idx, result = await future
            results[idx] = result

        return results
