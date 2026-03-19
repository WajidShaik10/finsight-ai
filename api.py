# api.py

import os
import shutil
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from quant_rag_agent.modules.agent import QuantAgent
from quant_rag_agent.modules.ingester import DocumentIngester

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="quant_rag_agent/static"), name="static")

chats = {}

class ChatRequest(BaseModel):
    question: str

class CreateChatRequest(BaseModel):
    name: str
    document: str = ""

class ChatMessageRequest(BaseModel):
    chat_id: str
    question: str

CHATS_FILE = "quant_rag_agent/data/chats.json"

def save_chats():
    data = {
        chat_id: {
            "id": v["id"],
            "name": v["name"],
            "document": v["document"],
            "created_at": v["created_at"],
            "collection_name": v["collection_name"],
            "history": v["agent"].history
        }
        for chat_id, v in chats.items()
    }
    with open(CHATS_FILE, "w") as f:
        import json
        json.dump(data, f, indent=2)

def load_chats():
    if not os.path.exists(CHATS_FILE):
        return {}
    try:
        import json
        with open(CHATS_FILE, "r") as f:
            data = json.load(f)
        loaded = {}
        for chat_id, v in data.items():
            collection_name = v.get("collection_name", f"chat_{chat_id.replace('-', '_')}")
            agent = QuantAgent(collection_name=collection_name)
            agent.history = v.get("history", [])
            ingester = DocumentIngester(collection_name=collection_name)
            loaded[chat_id] = {
                "id": v["id"],
                "name": v["name"],
                "document": v["document"],
                "created_at": v["created_at"],
                "collection_name": collection_name,
                "agent": agent,
                "ingester": ingester
            }
        print(f"Loaded {len(loaded)} chats from disk")
        return loaded
    except Exception as e:
        print(f"Error loading chats: {e}")
        return {}

chats = load_chats()
ingester = DocumentIngester()

@app.get("/")
def dashboard():
    return FileResponse("quant_rag_agent/static/chat.html")

@app.get("/old")
def old_dashboard():
    return FileResponse("quant_rag_agent/static/index.html")

@app.get("/documents")
def list_documents():
    files = [f for f in os.listdir("quant_rag_agent/data")
             if f.endswith(('.pdf', '.html'))]
    return {"documents": files}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = f"quant_rag_agent/data/{file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {"message": f"Saved {file.filename}", "filename": file.filename}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chats/create")
def create_chat(request: CreateChatRequest):
    chat_id = str(uuid.uuid4())
    collection_name = f"chat_{chat_id.replace('-', '_')}"
    chat_ingester = DocumentIngester(collection_name=collection_name)
    if request.document:
        file_path = f"quant_rag_agent/data/{request.document}"
        if os.path.exists(file_path):
            chat_ingester.ingest(file_path)
    chats[chat_id] = {
        "id": chat_id,
        "name": request.name,
        "document": request.document,
        "created_at": datetime.now().isoformat(),
        "collection_name": collection_name,
        "agent": QuantAgent(collection_name=collection_name),
        "ingester": chat_ingester
    }
    save_chats()
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
    answer = chats[request.chat_id]["agent"].ask(request.question)
    save_chats()
    return {"answer": answer}

@app.post("/chats/{chat_id}/upload")
async def upload_to_chat(chat_id: str, file: UploadFile = File(...)):
    try:
        if chat_id not in chats:
            return {"error": "Chat not found"}
        file_path = f"quant_rag_agent/data/{file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        chats[chat_id]["ingester"].ingest(file_path)
        chats[chat_id]["document"] = file.filename
        save_chats()
        return {"message": f"Successfully ingested {file.filename}", "filename": file.filename}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str):
    if chat_id in chats:
        del chats[chat_id]
        save_chats()
    return {"message": "Chat deleted"}

legacy_agent = QuantAgent()

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        answer = legacy_agent.ask(request.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}