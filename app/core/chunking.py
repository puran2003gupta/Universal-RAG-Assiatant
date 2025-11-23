# app/core/chunking.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
#A built-in LangChain utility used to split large text into smaller, meaningful chunks for embeddings.
# It intelligently breaks text at sentence or paragraph boundaries (not in the middle of a word or sentence).

from typing import List, Dict

# Used for type hinting — to make it clear that the function returns a list of dictionaries.
def chunk_text(raw_text: str, source: str, chunk_size=1200, overlap=200) -> List[Dict]:
    #chunk_size → how big each text chunk should be (default = 1200 characters).
    #Overlap is a crucial parameter in text chunking for maintaining contextual continuity, ensuring that semantically 
    # related information is not abruptly split across different chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, # Maximum length of each chunk
        chunk_overlap=overlap, # Keeps 200 characters from previous chunk to maintain context
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(raw_text)
    # split_text() is the function that actually performs the chunking.
    return [{"content": c, "metadata": {"source": source}} for c in chunks]