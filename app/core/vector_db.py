from langchain_community.vectorstores import Chroma

# Chroma is a local vector database from ChromaDB
# It stores and searches vector embeddings efficiently.
from app.config import CHROMA_DIR
from typing import List, Dict

def build_vectorstore(chunks: List[Dict], embedder):
    #persist_dir → where ChromaDB will save the data on disk (so you don’t lose it on restart).
    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    vs = Chroma.from_texts(
        texts=texts,
        embedding=embedder,
        metadatas=metadatas,
        persist_directory=CHROMA_DIR
    )
    vs.persist()
    return vs
