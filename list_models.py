# list_models.py
import os
from google import genai
from dotenv import load_dotenv

# Load .env file manually
load_dotenv()

print("GOOGLE_API_KEY =", bool(os.getenv("GOOGLE_API_KEY")))
print("GEMINI_API_KEY =", bool(os.getenv("GEMINI_API_KEY")))

# Prefer GOOGLE_API_KEY (used by SDK)
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ ERROR: No API key found in environment. Check your .env file.")
    exit()

# Pass API key correctly to the client
c = genai.Client(api_key=api_key)

print("\nListing available models:\n")
try:
    for m in c.models.list():
        print("-", m.name)
except Exception as e:
    print("Failed to list models:", e)
