# quant_rag_agent/modules/ingester.py

import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
import os
import re

class DocumentIngester:
    """
    Reads PDFs and HTML files, chunks the text, embeds it,
    and stores it in ChromaDB.
    """

    def __init__(self, db_path="./quant_rag_agent/db"):
        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="quant_documents"
        )
        print("ChromaDB ready!")

    def load_pdf(self, pdf_path, skip_pages=10):
        """Extract raw text from a PDF file, skipping intro pages."""
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
        """Extract raw text from an HTML file."""
        print(f"Reading {html_path}...")
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        # Remove all HTML tags
        clean = re.sub(r'<[^>]+>', ' ', html)
        # Remove extra whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean

    def chunk_text(self, text, chunk_size=1000, overlap=100):
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def ingest(self, file_path):
        """Full pipeline: file -> chunks -> embeddings -> ChromaDB."""

        # Step 1: Extract text based on file type
        if file_path.endswith('.pdf'):
            text = self.load_pdf(file_path)
        elif file_path.endswith('.html'):
            text = self.load_html(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        # Step 2: Chunk the text
        chunks = self.chunk_text(text)
        print(f"Created {len(chunks)} chunks")

        # Step 3: Embed all chunks
        print("Embedding chunks...")
        embeddings = self.model.encode(chunks).tolist()

        # Step 4: Store in ChromaDB
        doc_name = os.path.basename(file_path)
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"{doc_name}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"source": doc_name, "chunk": i} for i in range(len(chunks))]
        )
        print(f"✅ Ingested {doc_name} — {len(chunks)} chunks stored!")