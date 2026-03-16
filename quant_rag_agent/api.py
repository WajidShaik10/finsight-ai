# api.py

import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from quant_rag_agent.modules.agent import QuantAgent
from quant_rag_agent.modules.ingester import DocumentIngester

from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Quant RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = QuantAgent()
ingester = DocumentIngester()

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def dashboard():
    return FileResponse("quant_rag_agent/static/index.html")

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        answer = agent.ask(request.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = f"quant_rag_agent/data/{file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        ingester.ingest(file_path)
        return {"message": f"Successfully ingested {file.filename}"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/documents")
def list_documents():
    data_dir = "quant_rag_agent/data"
    files = [f for f in os.listdir(data_dir)
             if f.endswith('.pdf') or f.endswith('.html')]
    return {"documents": files}

app.mount("/static", StaticFiles(directory="quant_rag_agent/static"), name="static")