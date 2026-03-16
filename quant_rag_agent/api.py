# api.py

import os
import shutil
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from quant_rag_agent.modules.agent import QuantAgent
from quant_rag_agent.modules.ingester import DocumentIngester

load_dotenv()

app = FastAPI(title="Quant RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store chats in memory
chats = {}

ingester = DocumentIngester()

class ChatRequest(BaseModel):
    question: str

class CreateChatRequest(BaseModel):
    name: str
    document: str = ""

class ChatMessageRequest(BaseModel):
    chat_id: str
    question: str

# ─────────────────────────────────────────
# Pages
# ─────────────────────────────────────────

@app.get("/")
def dashboard():
    return FileResponse("quant_rag_agent/static/chat.html")

@app.get("/old")
def old_dashboard():
    return FileResponse("quant_rag_agent/static/index.html")

# ─────────────────────────────────────────
# Documents
# ─────────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = f"quant_rag_agent/data/{file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        ingester.ingest(file_path)
        return {"message": f"Successfully ingested {file.filename}", "filename": file.filename}
    except Exception as e:
        return {"error": str(e)}

@app.get("/documents")
def list_documents():
    files = [f for f in os.listdir("quant_rag_agent/data")
             if f.endswith(('.pdf', '.html'))]
    return {"documents": files}

# ─────────────────────────────────────────
# Chats
# ─────────────────────────────────────────

@app.post("/chats/create")
def create_chat(request: CreateChatRequest):
    chat_id = str(uuid.uuid4())
    chats[chat_id] = {
        "id": chat_id,
        "name": request.name,
        "document": request.document,
        "created_at": datetime.now().isoformat(),
        "agent": QuantAgent()
    }
    return {"chat_id": chat_id, "name": request.name}

@app.get("/chats")
def list_chats():
    return {"chats": [
        {
            "id": v["id"],
            "name": v["name"],
            "document": v["document"],
            "created_at": v["created_at"]
        }
        for v in chats.values()
    ]}

@app.post("/chats/message")
def chat_message(request: ChatMessageRequest):
    if request.chat_id not in chats:
        return {"error": "Chat not found"}
    chat = chats[request.chat_id]
    answer = chat["agent"].ask(request.question)
    return {"answer": answer}

@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str):
    if chat_id in chats:
        del chats[chat_id]
    return {"message": "Chat deleted"}

# ─────────────────────────────────────────
# Legacy single chat
# ─────────────────────────────────────────

legacy_agent = QuantAgent()

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        answer = legacy_agent.ask(request.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}