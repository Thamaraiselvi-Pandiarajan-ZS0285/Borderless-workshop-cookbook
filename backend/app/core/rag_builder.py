class RAGBuilder:
    def __init__(self):
        pass

    def build_context(self, ranked_chunks: list[dict]) -> str:
        return "\n---\n".join([chunk['content'] for chunk in ranked_chunks])
