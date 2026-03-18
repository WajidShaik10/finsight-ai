# quant_rag_agent/modules/ingester.py

import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
import os
import re

class DocumentIngester:
    def __init__(self, db_path="./quant_rag_agent/db", collection_name="quant_documents"):
        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        print("ChromaDB ready!")

    def load_pdf(self, pdf_path, skip_pages=0):
        print(f"Reading {pdf_path}...")
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i < skip_pages:
                    continue
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def load_html(self, html_path):
        print(f"Reading {html_path}...")
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        clean = re.sub(r'<[^>]+>', ' ', html)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean

    def chunk_text(self, text, chunk_size=1000, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def ingest(self, file_path):
        if file_path.endswith('.pdf'):
            text = self.load_pdf(file_path)
        elif file_path.endswith('.html'):
            text = self.load_html(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        chunks = self.chunk_text(text)
        if not chunks:
            raise ValueError("No text extracted from file")

        print(f"Created {len(chunks)} chunks")
        print("Embedding chunks...")
        embeddings = self.model.encode(chunks).tolist()

        doc_name = os.path.basename(file_path)
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"{doc_name}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"source": doc_name, "chunk": i} for i in range(len(chunks))]
        )
        print(f"Ingested {doc_name} — {len(chunks)} chunks stored!")