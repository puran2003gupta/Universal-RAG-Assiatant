
# app/web/streamlit_app.py
import streamlit as st
import requests
import json
from datetime import datetime
from io import BytesIO
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")  # adjust if needed

st.set_page_config(page_title="Universal RAG Assistant", page_icon="🤖", layout="wide")
st.title("🌐 Universal RAG Assistant")
st.write("Upload a PDF or enter a webpage URL, then ask questions based on the content.")

# ---------------------
# Session state init
# ---------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role":"user"/"assistant","content":..., "ts":...}
# Default chat name is now a friendly label rather than a timestamp
if "chat_name" not in st.session_state:
    st.session_state.chat_name = "Start Chatting"
if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = {}
# flag to clear the input on next render (avoids modifying session_state after widget exists)
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False
# If clear_input flag set, clear input_text BEFORE we create the text_input widget (safe).
if st.session_state.clear_input:
    st.session_state["input_text"] = ""
    st.session_state.clear_input = False

# ---------------------
# Sidebar: Ingest + Chat actions
# ---------------------
with st.sidebar:
    st.header("📥 Ingest Data")
    ingest_choice = st.radio("Choose input type:", ["Web URL", "PDF Upload"])
    if ingest_choice == "Web URL":
        url_input = st.text_input("Enter a webpage URL:", key="url_input")
        if st.button("Ingest URL"):
            if not url_input.strip():
                st.error("Please enter a valid URL.")
            else:
                with st.spinner("Extracting and processing text..."):
                    try:
                        response = requests.post(f"{BACKEND_URL}/ingest_url", data={"url": url_input}, timeout=60)
                        if response.status_code == 200:
                            st.success("✅ URL ingested successfully!")
                        else:
                            st.error(f"❌ Error: {response.status_code} {response.text}")
                    except Exception as e:
                        st.error(f"❌ Exception: {e}")

    else:
        pdf_file = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_uploader")
        if st.button("Ingest PDF"):
            if not pdf_file:
                st.error("Please upload a PDF file.")
            else:
                with st.spinner("Uploading and processing PDF..."):
                    try:
                        files = {"file": (pdf_file.name, pdf_file, "application/pdf")}
                        response = requests.post(f"{BACKEND_URL}/ingest_pdf", files=files, timeout=120)
                        if response.status_code == 200:
                            st.success("✅ PDF ingested successfully!")
                        else:
                            st.error(f"❌ Error: {response.status_code} {response.text}")
                    except Exception as e:
                        st.error(f"❌ Exception: {e}")

    st.markdown("---")
    st.header("💬 Chat Controls")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔄 New chat"):
            st.session_state.history = []
            st.session_state.chat_name = "Start Chatting"
            st.rerun()
    with col2:
        if st.button("💾 Quick save current chat"):
            name = st.session_state.chat_name or f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.saved_chats[name] = list(st.session_state.history)
            st.success(f"Saved as '{name}'")

    if st.session_state.saved_chats:
        st.markdown("**Saved chats**")
        sel = st.selectbox("Open saved chat:", ["(select)"] + list(st.session_state.saved_chats.keys()), key="open_saved")
        if sel and sel != "(select)":
            st.session_state.history = list(st.session_state.saved_chats[sel])
            st.session_state.chat_name = sel
            st.rerun()

    st.markdown("---")
    if st.button("📤 Download chat (.json)"):
        blob = json.dumps(st.session_state.history, indent=2, ensure_ascii=False)
        st.download_button("Download JSON", data=blob, file_name=f"{st.session_state.chat_name}.json")
    st.caption("Built with Code - Developer Puran 👨🏼‍💻 ")

# ---------------------
# Main chat area
# ---------------------
# Show a friendly header. If you want a different label, change chat_name value above.
st.header(f"💬 {st.session_state.chat_name}")

# display chat history
chat_box = st.container()
with chat_box:
    if not st.session_state.history:
        # Show a simple prompt instead of a timestamped "Chat XXX"
        st.info("Start Chatting — ask a question below.")
    else:
        for msg in st.session_state.history:
            role = msg.get("role")
            content = msg.get("content") or ""
            # remove timestamp display — keep ts stored but don't show it
            safe_content = content.replace("\n", "<br>")

            if role == "user":
                # user bubble — right aligned, dark text on light bubble
                st.markdown(
                    f"""
                    <div style="text-align:right; padding:6px; margin-bottom:8px;">
                      <div style="font-weight:bold; color:#ffffff; margin-bottom:6px;">You</div>
                      <div style="
                            display:inline-block;
                            background:#d1eaff;
                            color:#000000;
                            padding:12px;
                            border-radius:12px;
                            max-width:80%;
                            word-wrap:break-word;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
                          ">{safe_content}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # assistant bubble — left aligned, dark text on slightly different light bubble
                sources = msg.get("sources", [])
                st.markdown(
                    f"""
                    <div style="text-align:left; padding:6px; margin-bottom:8px;">
                      <div style="font-weight:bold; color:#ffffff; margin-bottom:6px;">Assistant</div>
                      <div style="
                            display:inline-block;
                            background:#d1eaff;
                            color:#000000;
                            padding:12px;
                            border-radius:12px;
                            max-width:80%;
                            word-wrap:break-word;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
                          ">{safe_content}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                

st.markdown("---")

# ---------------------
# Input area (safe clear logic)
# ---------------------
col_user, col_send = st.columns([9, 1])
with col_user:
    # Using key="input_text" ensures the value is stored in st.session_state["input_text"]
    user_query = st.text_input("Type your question:", key="input_text")
with col_send:
    send = st.button("Ask")

# We'll set this flag to True if we need to rerun after the try/except finishes.
_need_rerun_after_try = False

if send:
    q = (st.session_state.get("input_text") or "").strip()
    if not q:
        st.warning("Please enter a question.")
    else:
        user_msg = {"role": "user", "content": q, "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        st.session_state.history.append(user_msg)

        payload = {
            "q": q,
            "history": st.session_state.history,  # backend expects this shape
        }
        with st.spinner("Generating answer..."):
            try:
                response = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=120)
                if response.status_code == 200:
                    data = response.json()
                    answer_text = None
                    sources = []
                    if isinstance(data, dict):
                        if "answer" in data and isinstance(data["answer"], dict):
                            answer_text = data["answer"].get("answer") or data["answer"].get("text")
                            sources = data["answer"].get("sources", [])
                        elif "answer" in data and isinstance(data["answer"], str):
                            answer_text = data["answer"]
                        elif "text" in data:
                            answer_text = data["text"]
                        else:
                            answer_text = json.dumps(data)[:2000]
                    else:
                        answer_text = str(data)

                    if not answer_text:
                        answer_text = "No answer returned by backend."

                    assistant_msg = {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    st.session_state.history.append(assistant_msg)

                    # set the flag so that next render clears the input BEFORE widget instantiation
                    st.session_state.clear_input = True

                    # mark that we need to rerun, but do it after try/except ends
                    _need_rerun_after_try = True
                else:
                    st.error(f"Backend error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Exception while calling backend: {e}")

# After network try/except finishes, if we flagged a rerun, call st.rerun() outside try/except so it's not treated as an error.
if _need_rerun_after_try:
    st.rerun()


