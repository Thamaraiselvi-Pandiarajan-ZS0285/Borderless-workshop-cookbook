RERANK_PROMPT = (
    """You are given a user query and a list of retrieved passages.

    Your task is to:
    1. Assign a relevance score (from 1 to 100) to each passage based on its relevance to the query.
    2. Sort the passages by their score in descending order (most relevant first).
    3. Return the result in the format:
       Rank. [Score: X] <passage text>

    Query:
    {query}

    Passages:
    {passages}
    """
)