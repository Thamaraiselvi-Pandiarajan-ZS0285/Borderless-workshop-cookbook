import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")

def split_text(text, max_tokens=1024):
    words = text.split()
    chunks, current_chunk = [], []
    current_len = 0
    for word in words:
        word_tokens = len(ENC.encode(word))
        if current_len + word_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_len = word_tokens
        else:
            current_chunk.append(word)
            current_len += word_tokens
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
# todo add one more condition to embedd the chunk when size is less than 512