# app/agents/answer_agent.py
from typing import List, Dict
import os, traceback

from app.config import GEMINI_API_KEY, LLM_MODEL_NAME

# try to import new SDK
_genai_client = None
_genai_mode = None
try:
    from google import genai  # new SDK
    try:
        _genai_client = genai.Client()
        _genai_mode = "new"
    except Exception as e:
        _genai_client = None
        _genai_mode = None
        print(f"Warning: genai.Client() creation failed: {e}")
except Exception:
    _genai_client = None
    _genai_mode = None

# fallback older package already handled elsewhere; ensure _genai_mode set
if _genai_mode is None:
    try:
        import google.generativeai as old_genai  # type: ignore
        if GEMINI_API_KEY:
            try:
                old_genai.configure(api_key=GEMINI_API_KEY)
                _genai_client = old_genai
                _genai_mode = "old"
            except Exception as e:
                print(f"Warning: configure old genai failed: {e}")
    except Exception:
        _genai_mode = None
        _genai_client = None

if _genai_mode is None:
    print("WARNING: No Google GenAI SDK available. Install google-genai or google-generativeai.")

def build_prompt(query: str, chunks: List[Dict]) -> str:
    context_parts = []
    for i, c in enumerate(chunks, start=1):
        src = c["metadata"].get("source", f"chunk_{i}")
        excerpt = c["content"][:800].replace("\n", " ")
        context_parts.append(f"Source {i} ({src}):\n{excerpt}")
    context_text = "\n\n".join(context_parts)
    system = (
        "You are an assistant that answers queries using the provided sources. "
        "Cite the source number(s) you used at the end of the answer in square brackets."
    )
    return f"{system}\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{query}\n\nAnswer concisely and include citations."

def _list_available_models_new_sdk():
    """Return a short list of available models from new SDK (or throwable message)."""
    try:
        # new SDK: client.models.list()
        models = _genai_client.models.list().models
        return [m.name for m in models]
    except Exception as e:
        return f"Failed to list models: {e}"

def _generate_with_new_sdk(prompt: str, model: str):
    """Call new SDK generate_content but raise helpful error on 404."""
    try:
        response = _genai_client.models.generate_content(model=model, contents=prompt)
        # Try common return fields
        if hasattr(response, "text") and response.text:
            return response.text
        try:
            return response.candidates[0].content[0].text
        except Exception:
            return str(response)
    except Exception as e:
        # If API reported 404 or model error, try list_models and surface friendly message
        err = e
        try:
            models = _list_available_models_new_sdk()
        except Exception:
            models = f"Could not fetch models list: {traceback.format_exc()}"
        raise RuntimeError(
            f"GenAI generation failed for model '{model}': {err}\n\n"
            f"Available models (partial): {models}\n\n"
            "Please choose an available model name and set LLM_MODEL_NAME in your .env. "
            "Common options (if available in your account) might be: 'text-bison-001', 'gemini-pro', or other listed above."
        )

def _generate_with_old_sdk(prompt: str, model: str):
    try:
        if hasattr(_genai_client, "generate"):
            resp = _genai_client.generate(model=model, input=prompt, temperature=0.0, max_output_tokens=512)
            return getattr(resp, "text", str(resp))
        if hasattr(_genai_client, "generate_content"):
            resp = _genai_client.generate_content(model=model, contents=prompt)
            return getattr(resp, "text", str(resp))
        raise RuntimeError("Old SDK has no supported generation method.")
    except Exception as e:
        raise RuntimeError(f"Old SDK generation failed: {e}\n{traceback.format_exc()}")

def generate_answer(query: str, chunks: List[Dict]):
    prompt = build_prompt(query, chunks)
    model_name = os.getenv("LLM_MODEL_NAME", LLM_MODEL_NAME)
    if _genai_mode == "new":
        text = _generate_with_new_sdk(prompt, model=model_name)
    elif _genai_mode == "old":
        text = _generate_with_old_sdk(prompt, model=model_name)
    else:
        # No SDK: fallback to local summarization for dev testing
        summary = "\n\n".join(c["content"][:500] for c in chunks[:3])
        return {"answer": f"[LOCAL-FALLBACK] {summary}", "sources": [c["metadata"].get("source") for c in chunks]}
    return {"answer": text, "sources": [c["metadata"].get("source") for c in chunks]}
