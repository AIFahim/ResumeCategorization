from fastembed import TextEmbedding
from langchain_text_splitters import TokenTextSplitter
from typing import List

class TextEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", chunk_size: int = 128, chunk_overlap: int = 10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = TextEmbedding(model_name=model_name)
        print(f"The model {model_name} is ready to use.")

    def chunk_text(self, text: str) -> List[str]:
        splitter = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = splitter.split_text(text)
        return chunks

    def calculate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        embeddings = list(self.embedding_model.embed(chunks))
        return embeddings

    def combine_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        combined_embedding = [sum(x) / len(x) for x in zip(*embeddings)]
        return combined_embedding

    def embed_text(self, text: str) -> List[float]:
        chunks = self.chunk_text(text)
        embeddings = self.calculate_embeddings(chunks)
        combined_embedding = self.combine_embeddings(embeddings)
        return combined_embedding

# Usage example:
if __name__ == "__main__":
    text = "This is an example document that needs embedding."
    embedder = TextEmbedder(model_name='BAAI/bge-small-en-v1.5', chunk_size=128, chunk_overlap=10)
    combined_embedding = embedder.embed_text(text)
    print(f"Combined embedding of the document: {combined_embedding}")
    print(f"Length of the embedding vector: {len(combined_embedding)}")
