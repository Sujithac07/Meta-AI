"""
Production Streamlit UI with chat history, file upload, and RAG support.
"""
import streamlit as st
import requests
import uuid
import os

st.set_page_config(
    page_title="Meta AI: An Intelligent System for Automated Machine Learning Pipeline Generation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration from environment or defaults
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
    }
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    .stChatMessage {
        border-radius: 15px;
        margin-bottom: 10px;
    }
    .st-emotion-cache-1c79332 {
        background-color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ META-AI Settings")
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
    
    st.divider()
    st.subheader("📄 Knowledge Base (RAG)")
    uploaded_file = st.file_uploader(
        "Upload a document for AI to reference",
        type=["pdf", "txt", "docx", "csv", "md"],
        help="AI will use this document to answer your questions"
    )
    
    if uploaded_file:
        with st.spinner("Indexing document..."):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            try:
                response = requests.post(
                    f"{API_URL}/documents/ingest",
                    files=files,
                    timeout=30,
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success(
                        f"✅ Indexed {data['chunks_created']} chunks from '{uploaded_file.name}'"
                    )
                    st.session_state.rag_enabled = True
                else:
                    st.error(f"Failed to index. Server returned: {response.status_code}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
    
    if st.session_state.rag_enabled:
        st.info("🔍 RAG Active — AI is using your knowledge base")
    
    st.divider()
    st.subheader("🎨 Creative AI")
    with st.expander("Generate Image"):
        img_prompt = st.text_area("Image Description", placeholder="A futuristic laboratory with AI robots...")
        if st.button("Generate Image"):
            with st.spinner("Meta AI is painting..."):
                try:
                    res = requests.post(
                        f"{API_URL}/images/generate", 
                        json={"session_id": st.session_state.session_id, "prompt": img_prompt},
                        timeout=30,
                    )
                    if res.status_code == 200:
                        data = res.json()
                        st.image(data["image_url"], caption=img_prompt)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"![{img_prompt}]({data['image_url']})"
                        })
                    else:
                        st.error("Failed to generate image.")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.subheader("🎛️ Options")
    streaming = st.toggle("⚡ Stream responses", value=True)
    show_sources = st.toggle("📚 Show sources", value=True)
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        try:
            requests.delete(f"{API_URL}/sessions/{st.session_state.session_id}", timeout=15)
        except Exception:
            st.warning("Could not clear server-side session. Local chat was cleared.")
        st.rerun()

# ── Main Chat Interface ───────────────────────────────
st.title("🤖 META-AI")
st.caption("Production-Grade AI Assistant • RAG • Persistent Memory")

# Display chat history from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if show_sources and message.get("sources"):
            with st.expander("📚 Sources"):
                st.text(message["sources"])

# Chat input
if prompt := st.chat_input("Ask anything..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        if streaming:
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                with requests.post(
                    f"{API_URL}/chat/stream",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": prompt,
                        "rag_enabled": st.session_state.rag_enabled
                    },
                    stream=True,
                    timeout=60
                ) as response:
                    for chunk in response.iter_content(chunk_size=None):
                        if chunk:
                            token = chunk.decode("utf-8")
                            full_response += token
                            response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })
            except Exception as e:
                st.error(f"Streaming failed: {e}")
        else:
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "session_id": st.session_state.session_id,
                            "message": prompt,
                            "rag_enabled": st.session_state.rag_enabled
                        },
                        timeout=60
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.markdown(data["message"])
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": data["message"],
                            "sources": data.get("sources")
                        })
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
