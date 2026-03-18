# quant_rag_agent/modules/retriever.py

import chromadb
from sentence_transformers import SentenceTransformer

class DocumentRetriever:
    def __init__(self, db_path="./quant_rag_agent/db", collection_name="quant_documents"):
        print("Loading retriever...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        print("Retriever ready!")

    def retrieve(self, query, top_k=5):
        query_embedding = self.model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        chunks = results["documents"][0]
        sources = results["metadatas"][0]
        print(f"\n🔍 Query: {query}")
        print(f"📄 Found {len(chunks)} relevant chunks\n")
        for i, (chunk, source) in enumerate(zip(chunks, sources)):
            print(f"--- Chunk {i+1} (from {source['source']}) ---")
            print(chunk[:200])
            print()
        return chunks