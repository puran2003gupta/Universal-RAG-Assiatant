# app/config.py
import os
from dotenv import load_dotenv

# load .env (if present)
load_dotenv()

# API keys / model names
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")

# Backwards-compatible aliases used across the codebase
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", MODEL_NAME)
EMBED_MODEL = os.getenv("EMBED_MODEL", os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2"))

# Storage / misc
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")
USER_AGENT = os.getenv("USER_AGENT", "UniversalRagAssistant/1.0")

# Dev-friendly: warn (don't crash) if key missing
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set. LLM calls will fail until you set it in .env.")
