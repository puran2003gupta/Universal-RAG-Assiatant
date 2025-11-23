from langchain_community.embeddings import HuggingFaceEmbeddings
#This imports the HuggingFaceEmbeddings class from LangChain.
# It wraps sentence-transformer models from Hugging Face into an easy-to-use format for LangChain pipelines.
# These models are designed to convert text into semantic vectors — meaning they capture context and meaning, not just raw words.
def build_embedder(model_name: str = "all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

#model_name: specifies which Hugging Face model to use for embeddings.
#Default → "all-MiniLM-L6-v2", a lightweight, high-quality sentence transformer.
