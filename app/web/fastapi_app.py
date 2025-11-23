# app/web/fastapi_app.py
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime
import logging
import os

# import your existing modules (exact names you provided)
from app.agents.ingestion_agent import extract_pdf_text, extract_web_text
from app.core.chunking import chunk_text
from app.core.embedding import build_embedder
from app.core.vector_db import build_vectorstore
from app.agents.retrieval_agent import retrieve_relevant_chunks
from app.agents.answer_agent import generate_answer
from app.config import CHROMA_DIR, EMBED_MODEL  # keep if used elsewhere

# ---- Config ----
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", 8))
MAX_HISTORY_CHAR = int(os.environ.get("MAX_HISTORY_CHAR", 400))
RETRIEVE_K = int(os.environ.get("RETRIEVE_K", 4))

logger = logging.getLogger("uvicorn.error")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory conversation store (simple). For production persist to DB.
_conversation_store: Dict[str, Dict[str, Any]] = {}

# Pydantic models for request bodies
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    ts: Optional[str] = None

class AskPayload(BaseModel):
    q: str
    history: Optional[List[Message]] = []
    conversation_id: Optional[str] = None

class SaveChatPayload(BaseModel):
    name: Optional[str] = None
    history: List[Message]

# -----------------------
# Ingest endpoints (unchanged behavior)
# -----------------------
@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile):
    path = f"data/pdfs/{file.filename}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(await file.read())
    text, pages = extract_pdf_text(path)
    chunks = chunk_text(text, source=f"PDF: {file.filename}")
    embedder = build_embedder()
    vs = build_vectorstore(chunks, embedder)  # uses CHROMA_DIR from config
    return {"status": "ok", "chunks": len(chunks)}

@app.post("/ingest_url")
async def ingest_url(url: str = Form(...)):
    text = extract_web_text(url)
    chunks = chunk_text(text, source=f"URL: {url}")
    embedder = build_embedder()
    vs = build_vectorstore(chunks, embedder)
    return {"status": "ok", "chunks": len(chunks)}

# -----------------------
# Conversational ask: POST (preferred)
# -----------------------
@app.post("/ask")
async def ask_post(payload: AskPayload):
    """
    Expects JSON:
    {
      "q": "current question",
      "history": [ {"role":"user"/"assistant","content":"...","ts":"..."} ],
      "conversation_id": "optional-uuid"
    }
    Returns: {"answer": {"answer": "...", "sources": [...]}}
    """
    q = payload.q
    history = payload.history or []
    conv_id = payload.conversation_id

    # 1) Build compact transcript from last messages
    try:
        trimmed = history[-MAX_HISTORY_MESSAGES:]
        lines = []
        for m in trimmed:
            role = (m.role or "").lower()
            content = (m.content or "")
            if len(content) > MAX_HISTORY_CHAR:
                content = content[:MAX_HISTORY_CHAR] + "...(truncated)"
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {content}")
        conversation_text = "\n".join(lines)
    except Exception as e:
        logger.exception("Error preparing history")
        conversation_text = ""

    # 2) Build augmented query (history + current question)
    if conversation_text:
        augmented_query = f"Conversation history:\n{conversation_text}\n\nCurrent question:\n{q}"
    else:
        augmented_query = q

    # 3) Retrieve relevant chunks for the current question (RAG)
    try:
        retrieved_chunks = retrieve_relevant_chunks(q, k=RETRIEVE_K)
    except Exception as e:
        logger.exception("Retrieval failed")
        retrieved_chunks = []

    # 4) Call generate_answer(augmented_query, chunks)
    try:
        answer_obj = generate_answer(augmented_query, retrieved_chunks)
    except Exception as e:
        logger.exception("generate_answer failed")
        raise HTTPException(status_code=500, detail=f"generate_answer error: {e}")

    # 5) Normalize output to { "answer": "<text>", "sources": [...] }
    if isinstance(answer_obj, dict):
        answer_text = answer_obj.get("answer") or answer_obj.get("text") or str(answer_obj)
        sources = answer_obj.get("sources", [])
    else:
        # If generate_answer returned a plain string
        answer_text = str(answer_obj)
        sources = []

    # 6) If no sources returned, derive from retrieved_chunks
    if (not sources) and retrieved_chunks:
        derived = []
        for c in retrieved_chunks:
            if isinstance(c, dict):
                src = c.get("metadata", {}).get("source") or c.get("metadata", {}).get("source_name") or str(c.get("score", "")) or str(c)
            else:
                src = str(c)
            derived.append(src)
        sources = derived

    # 7) If conversation_id provided, optionally append to in-memory store
    if conv_id:
        _conversation_store.setdefault(conv_id, {"name": f"conv_{conv_id}", "history": []})
        ts = datetime.utcnow().isoformat()
        _conversation_store[conv_id]["history"].append({"role": "user", "content": q, "ts": ts})
        _conversation_store[conv_id]["history"].append({"role": "assistant", "content": answer_text, "ts": ts})

    return {"answer": {"answer": answer_text, "sources": sources}}

# -----------------------
# Backwards-compatible GET /ask?q=...
# -----------------------
@app.get("/ask")
async def ask_get(q: str):
    """
    Backwards compatible GET ask endpoint.
    """
    try:
        chunks = retrieve_relevant_chunks(q, k=RETRIEVE_K)
    except Exception as e:
        logger.exception("Retrieval failed")
        chunks = []

    try:
        answer_obj = generate_answer(q, chunks)
    except Exception as e:
        logger.exception("generate_answer failed on GET")
        raise HTTPException(status_code=500, detail=f"generate_answer error: {e}")

    if isinstance(answer_obj, dict):
        answer_text = answer_obj.get("answer") or answer_obj.get("text") or str(answer_obj)
        sources = answer_obj.get("sources", [])
    else:
        answer_text = str(answer_obj)
        sources = []

    if (not sources) and chunks:
        derived = []
        for c in chunks:
            if isinstance(c, dict):
                src = c.get("metadata", {}).get("source") or c.get("metadata", {}).get("source_name") or str(c.get("score", "")) or str(c)
            else:
                src = str(c)
            derived.append(src)
        sources = derived

    return {"answer": {"answer": answer_text, "sources": sources}}

# -----------------------
# Optional: save/load chats (in-memory)
# -----------------------
@app.post("/save_chat")
async def save_chat(payload: SaveChatPayload):
    conv_id = str(uuid4())
    name = payload.name or f"chat_{conv_id[:8]}"
    _conversation_store[conv_id] = {"name": name, "history": [m.dict() for m in payload.history]}
    return {"conversation_id": conv_id, "name": name}

@app.get("/load_chat")
async def load_chat(conversation_id: str):
    conv = _conversation_store.get(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="conversation_id not found")
    return {"conversation_id": conversation_id, "name": conv.get("name"), "history": conv.get("history", [])}


@app.get("/")
async def health():
    return {
        "status": "ok",
        "message": "Universal RAG Assistant backend is running. See /docs for API.",
    }


