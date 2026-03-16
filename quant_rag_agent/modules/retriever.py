# quant_rag_agent/modules/retriever.py

import chromadb
from sentence_transformers import SentenceTransformer

class DocumentRetriever:
    """
    Searches ChromaDB for the most relevant chunks
    based on a user query.
    """

    def __init__(self, db_path="./quant_rag_agent/db"):
        # Load the same embedding model we used in ingester
        print("Loading retriever...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Connect to the same ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)

        # Load the same collection we stored chunks in
        self.collection = self.client.get_or_create_collection(
            name="quant_documents"
        )
        print("Retriever ready!")

    def retrieve(self, query, top_k=3):
        """
        Takes a question, finds the most relevant chunks.

        query  = your question e.g. "What is RAG?"
        top_k  = how many chunks to return (default 3)
        """

        # Step 1: Convert the query into a vector
        query_embedding = self.model.encode(query).tolist()

        # Step 2: Search ChromaDB for closest chunks
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Step 3: Return chunks cleanly
        chunks = results["documents"][0]
        sources = results["metadatas"][0]

        print(f"\n🔍 Query: {query}")
        print(f"📄 Found {len(chunks)} relevant chunks:\n")

        for i, (chunk, source) in enumerate(zip(chunks, sources)):
            print(f"--- Chunk {i+1} (from {source['source']}) ---")
            print(chunk)
            print()

        return chunks