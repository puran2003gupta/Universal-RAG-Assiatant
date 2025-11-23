# app/agents/ingestion_agent.py
import fitz # fitz is part of PyMuPDF, a fast and layout-aware library for reading PDF content.
from typing import Tuple # The function’s return type is a tuple (str, int) → meaning it returns (text, number_of_pages).
from langchain_community.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup


def extract_pdf_text(path: str) -> Tuple[str, int]:
   
    try:
        doc = fitz.open(path) #Attempts to open the PDF file at path. fitz.open()
    except Exception as e:
        raise ValueError(f"Failed to open PDF '{path}': {e}")

    pages = []
    for page in doc:
        # "text" uses layout-aware extraction; fallback to "plain" if needed.
        content = page.get_text("text") or page.get_text()
        pages.append(content.strip()) #strip() removes leading/trailing whitespace
    doc.close()

    text = "\n\n".join(pages) #Joins all pages’ text with two line breaks
    return text, len(pages)



def extract_web_text(url: str) -> str:
    """
    Fetches textual content from a URL.
    Tries LangChain WebBaseLoader; falls back to requests+BeautifulSoup.
    """
    # ✅ Primary Method: LangChain WebBaseLoader
    try:    
        docs = WebBaseLoader(url).load() # WebBaseLoader is a built-in tool from LangChain.It automatically fetches and cleans text content from a webpage URL
        if docs:
            return "\n\n".join(d.page_content.strip() for d in docs if d.page_content)
    except Exception:
        pass  # fallback to manual extraction

    # ✅ Secondary Method: requests + BeautifulSoup
    #It downloads the raw HTML of the webpage.
    # Example: If you go to a website and view → “Page Source” — that’s what requests gets.
    # The User-Agent header helps pretend to be a browser so websites don’t block the request.

    # BeautifulSoup is a Python library used for parsing HTML.
    # It allows you to navigate and extract parts of a web page easily.
    # Think of it as a “web page cleaner” that removes unnecessary HTML tags.


    try:
        res = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        
        # Remove unnecessary tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Extract text content
        text = soup.get_text(separator="\n")

        # Clean up extra spaces and blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    except Exception as e:
        raise ValueError(f"Failed to fetch URL '{url}': {e}")   