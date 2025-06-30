class SummarizeAgent:
    def __init__(self, client):
        self.client = client

    async def summarize(self, content: str) -> str:
        prompt = f"Summarize the following text:\n\n{content}"
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": prompt}
        ]
        response = await self.client.acall(messages=messages)
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")
