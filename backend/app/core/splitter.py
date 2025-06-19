from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentSplitter:
    def __init__(self, chunk_size=512, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split(self, text: str) -> list[str]:
        return self.splitter.split_text(text)
