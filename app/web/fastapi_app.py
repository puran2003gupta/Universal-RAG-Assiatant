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


# # app/web/fastapi_app.py
# import os
# import sqlite3
# import json
# from typing import List, Dict, Any, Optional
# from uuid import uuid4
# from datetime import datetime
# import logging

# from fastapi import FastAPI, UploadFile, Form, HTTPException, Query
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel

# # ---- Adjust these imports to match your project layout ----
# from app.agents.ingestion_agent import extract_pdf_text, extract_web_text
# from app.core.chunking import chunk_text
# from app.core.embedding import build_embedder
# from app.core.vector_db import build_vectorstore
# from app.agents.retrieval_agent import retrieve_relevant_chunks
# from app.agents.answer_agent import generate_answer
# from app.config import CHROMA_DIR, EMBED_MODEL  # keep if used elsewhere
# # -----------------------------------------------------------

# logger = logging.getLogger("uvicorn.error")

# app = FastAPI()
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# # --- DB setup (SQLite) ---
# DB_DIR = os.environ.get("DATA_DIR", "data")
# DB_PATH = os.path.join(DB_DIR, "chats.db")
# os.makedirs(DB_DIR, exist_ok=True)

# def get_db_conn():
#     # check_same_thread=False allows access from multiple threads (uvicorn workers)
#     conn = sqlite3.connect(DB_PATH, check_same_thread=False)
#     conn.row_factory = sqlite3.Row
#     return conn

# # Create table if not exists
# _conn = get_db_conn()
# _cur = _conn.cursor()
# _cur.execute("""
# CREATE TABLE IF NOT EXISTS conversations (
#     id TEXT PRIMARY KEY,
#     name TEXT,
#     created_at TEXT,
#     updated_at TEXT,
#     history TEXT
# )
# """)
# _conn.commit()
# _conn.close()

# # Mount /local_files to serve files from /mnt/data (useful for debug images)
# LOCAL_FILES_DIR = os.environ.get("LOCAL_FILES_DIR", "/mnt/data")
# if os.path.exists(LOCAL_FILES_DIR):
#     app.mount("/local_files", StaticFiles(directory=LOCAL_FILES_DIR), name="local_files")


# # -----------------------
# # Pydantic models
# # -----------------------
# class Message(BaseModel):
#     role: str
#     content: str
#     ts: Optional[str] = None

# class AskPayload(BaseModel):
#     q: str
#     history: Optional[List[Message]] = []
#     conversation_id: Optional[str] = None

# class SaveChatPayload(BaseModel):
#     name: Optional[str] = None
#     history: List[Message]
#     conversation_id: Optional[str] = None  # optional: update existing if provided


# # -----------------------
# # Ingest endpoints (unchanged logic)
# # -----------------------
# @app.post("/ingest_pdf")
# async def ingest_pdf(file: UploadFile):
#     path = f"{DB_DIR}/pdfs/{file.filename}"
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, "wb") as f:
#         f.write(await file.read())
#     text, pages = extract_pdf_text(path)
#     chunks = chunk_text(text, source=f"PDF: {file.filename}")
#     embedder = build_embedder()
#     vs = build_vectorstore(chunks, embedder)
#     return {"status": "ok", "chunks": len(chunks)}

# @app.post("/ingest_url")
# async def ingest_url(url: str = Form(...)):
#     text = extract_web_text(url)
#     chunks = chunk_text(text, source=f"URL: {url}")
#     embedder = build_embedder()
#     vs = build_vectorstore(chunks, embedder)
#     return {"status": "ok", "chunks": len(chunks)}


# # -----------------------
# # Ask endpoints
# # -----------------------
# MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", 8))
# MAX_HISTORY_CHAR = int(os.environ.get("MAX_HISTORY_CHAR", 400))
# RETRIEVE_K = int(os.environ.get("RETRIEVE_K", 4))

# @app.post("/ask")
# async def ask_post(payload: AskPayload):
#     """
#     Conversational ask: accepts { q, history, conversation_id(optional) }.
#     Returns: {"answer": {"answer": "<text>", "sources": [...]}}
#     If conversation_id provided, append the user & assistant messages to stored conversation.
#     """
#     q = payload.q
#     history = payload.history or []
#     conv_id = payload.conversation_id

#     # Build compact transcript from last MAX_HISTORY_MESSAGES
#     try:
#         trimmed = history[-MAX_HISTORY_MESSAGES:]
#         lines = []
#         for m in trimmed:
#             role = (m.role or "").lower()
#             content = (m.content or "")
#             if len(content) > MAX_HISTORY_CHAR:
#                 content = content[:MAX_HISTORY_CHAR] + "...(truncated)"
#             prefix = "User" if role == "user" else "Assistant"
#             lines.append(f"{prefix}: {content}")
#         conversation_text = "\n".join(lines)
#     except Exception:
#         conversation_text = ""

#     augmented_query = f"Conversation history:\n{conversation_text}\n\nCurrent question:\n{q}" if conversation_text else q

#     # retrieval uses the current question
#     try:
#         retrieved_chunks = retrieve_relevant_chunks(q, k=RETRIEVE_K)
#     except Exception as e:
#         logger.exception("Retrieval failed")
#         retrieved_chunks = []

#     # generate answer using your existing answer_agent.generate_answer
#     try:
#         answer_obj = generate_answer(augmented_query, retrieved_chunks)
#     except Exception as e:
#         logger.exception("generate_answer failed")
#         raise HTTPException(status_code=500, detail=str(e))

#     # normalize the answer object
#     if isinstance(answer_obj, dict):
#         answer_text = answer_obj.get("answer") or answer_obj.get("text") or str(answer_obj)
#         sources = answer_obj.get("sources", [])
#     else:
#         answer_text = str(answer_obj)
#         sources = []

#     # derive sources if empty
#     if (not sources) and retrieved_chunks:
#         derived = []
#         for c in retrieved_chunks:
#             if isinstance(c, dict):
#                 src = c.get("metadata", {}).get("source") or str(c.get("score", "")) or str(c)
#             else:
#                 src = str(c)
#             derived.append(src)
#         sources = derived

#     # If conversation_id provided, append to stored conversation (create if not exists)
#     if conv_id:
#         conn = get_db_conn()
#         cur = conn.cursor()
#         now = datetime.utcnow().isoformat()
#         cur.execute("SELECT history, created_at FROM conversations WHERE id = ?", (conv_id,))
#         row = cur.fetchone()
#         if row:
#             try:
#                 existing = json.loads(row["history"])
#             except Exception:
#                 existing = []
#             existing.append({"role":"user","content": q, "ts": now})
#             existing.append({"role":"assistant","content": answer_text, "ts": now})
#             cur.execute("""
#                 UPDATE conversations
#                 SET history = ?, updated_at = ?
#                 WHERE id = ?
#             """, (json.dumps(existing), now, conv_id))
#         else:
#             # create new conversation
#             conv_name = f"chat_{conv_id[:8]}"
#             hist = [{"role":"user","content": q, "ts": now}, {"role":"assistant","content": answer_text, "ts": now}]
#             cur.execute("""
#                 INSERT INTO conversations (id, name, created_at, updated_at, history)
#                 VALUES (?, ?, ?, ?, ?)
#             """, (conv_id, conv_name, now, now, json.dumps(hist)))
#         conn.commit()
#         conn.close()

#     return {"answer": {"answer": answer_text, "sources": sources}}


# @app.get("/ask")
# async def ask_get(q: str = Query(...)):
#     """
#     Backwards-compatible GET /ask?q=...
#     """
#     try:
#         chunks = retrieve_relevant_chunks(q, k=RETRIEVE_K)
#     except Exception:
#         chunks = []

#     try:
#         answer_obj = generate_answer(q, chunks)
#     except Exception as e:
#         logger.exception("generate_answer failed (GET)")
#         raise HTTPException(status_code=500, detail=str(e))

#     if isinstance(answer_obj, dict):
#         answer_text = answer_obj.get("answer") or answer_obj.get("text") or str(answer_obj)
#         sources = answer_obj.get("sources", [])
#     else:
#         answer_text = str(answer_obj)
#         sources = []

#     if (not sources) and chunks:
#         derived = []
#         for c in chunks:
#             if isinstance(c, dict):
#                 src = c.get("metadata", {}).get("source") or str(c.get("score","")) or str(c)
#             else:
#                 src = str(c)
#             derived.append(src)
#         sources = derived

#     return {"answer": {"answer": answer_text, "sources": sources}}


# # -----------------------
# # Persistence endpoints (SQLite-backed)
# # -----------------------
# @app.post("/save_chat")
# async def save_chat_backend(payload: SaveChatPayload):
#     """
#     Save a chat. If payload.conversation_id is provided and exists -> update.
#     Returns conversation_id and name.
#     """
#     conn = get_db_conn()
#     cur = conn.cursor()
#     now = datetime.utcnow().isoformat()

#     conv_id = payload.conversation_id or str(uuid4())
#     name = payload.name or f"chat_{conv_id[:8]}"
#     try:
#         history_json = json.dumps([m.dict() for m in payload.history])
#     except Exception:
#         # fallback: try to serialize plain dicts
#         history_json = json.dumps(payload.history)

#     # check if exists
#     cur.execute("SELECT id FROM conversations WHERE id = ?", (conv_id,))
#     if cur.fetchone():
#         cur.execute("""
#             UPDATE conversations
#             SET name = ?, history = ?, updated_at = ?
#             WHERE id = ?
#         """, (name, history_json, now, conv_id))
#     else:
#         cur.execute("""
#             INSERT INTO conversations (id, name, created_at, updated_at, history)
#             VALUES (?, ?, ?, ?, ?)
#         """, (conv_id, name, now, now, history_json))
#     conn.commit()
#     conn.close()
#     return {"conversation_id": conv_id, "name": name}


# @app.get("/list_chats")
# async def list_chats():
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute("SELECT id, name, created_at, updated_at FROM conversations ORDER BY updated_at DESC")
#     rows = cur.fetchall()
#     conn.close()
#     items = [{"id": r["id"], "name": r["name"], "created_at": r["created_at"], "updated_at": r["updated_at"]} for r in rows]
#     return {"chats": items}


# @app.get("/load_chat")
# async def load_chat_backend(conversation_id: str = Query(...)):
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute("SELECT id, name, created_at, updated_at, history FROM conversations WHERE id = ?", (conversation_id,))
#     row = cur.fetchone()
#     conn.close()
#     if not row:
#         raise HTTPException(status_code=404, detail="conversation_id not found")
#     try:
#         history = json.loads(row["history"])
#     except Exception:
#         history = []
#     return {"conversation_id": row["id"], "name": row["name"], "history": history}


# @app.delete("/delete_chat")
# async def delete_chat_backend(conversation_id: str = Query(...)):
#     conn = get_db_conn()
#     cur = conn.cursor()
#     cur.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
#     conn.commit()
#     conn.close()
#     return {"deleted": conversation_id}
