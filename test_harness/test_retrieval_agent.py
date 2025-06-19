from backend.app.core.retrieval_agent import RetrievalAgent

agent = RetrievalAgent()
query = "Get all healthcare RFPs for India and summarize last month’s win rate?"
answer = agent.answer(query)
print(answer)
