# app/agents/retrieval_agent.py
from typing import List, Dict
from app.config import CHROMA_DIR
from app.core.embedding import build_embedder
# Use langchain_community import in codebase if available

from langchain_chroma import Chroma

EMBED_MODEL = None  # optional override if you want to pass

def _load_vectorstore(embedder=None):
    """
    Loads Chroma from disk. If embedder not provided, build default one.
    """
    if embedder is None:
        embedder = build_embedder()
    vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedder)
    return vs

def retrieve_relevant_chunks(query: str, k: int = 4) -> List[Dict]:
    """
    Returns top-k chunks for the query. Each item: {'content', 'metadata', 'score'}.
    """
    embedder = build_embedder()
    vs = _load_vectorstore(embedder=embedder)
    docs = vs.similarity_search_with_relevance_scores(query, k=k)  # yields (doc, score) pairs
    results = []
    for doc, score in docs:
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score)
        })
    return results
