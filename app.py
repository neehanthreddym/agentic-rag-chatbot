"""
Streamlit Chat UI — Agentic RAG Chatbot with Memory.

Showcases Feature A (RAG with citations) and Feature B (persistent memory)
in a polished, interactive web interface.

Run:  streamlit run app.py
"""

import os
import tempfile
import streamlit as st

# ----- Page config --------------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----- Custom CSS --------------------
st.markdown(
    """
<style>
/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* ---------- chat messages ---------- */
div[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.5rem;
}

/* ---------- expander (citations) ---------- */
details {
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
    margin-top: 0.5rem;
}
details summary {
    font-weight: 600;
}

/* ---------- file uploader ---------- */
section[data-testid="stFileUploader"] {
    border: 2px dashed rgba(108,99,255,0.5) !important;
    border-radius: 8px;
    padding: 0.75rem;
}

/* ---------- status badge ---------- */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-success { background: rgba(0,208,132,0.15); color: #00d084; }
.badge-muted   { background: rgba(138,143,152,0.12); color: #8a8f98; }

/* ---------- misc ---------- */
.block-container { max-width: 900px; }
</style>
""",
    unsafe_allow_html=True,
)


# ----- Session state defaults ----------
def _init_state():
    defaults = {
        "messages": [],
        "vectorstore": None,
        "ingested": False,
        "ingested_filename": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ----- Helper: read memory file ----------
import re

def _read_memory_file(path: str) -> str:
    """Return the content of a memory markdown file, stripping HTML comments."""
    if not os.path.exists(path):
        return ""
    with open(path, "r") as f:
        content = f.read()
    # Strip multi-line HTML comments  <!-- ... -->
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
    return content.strip()


# ----- Sidebar ----------
with st.sidebar:
    st.markdown("## 🤖 Chatbot")
    st.caption("Upload a PDF, then chat with it.")

    st.divider()

    # ----- PDF upload ----------
    st.markdown("### 📄 Document")
    uploaded = st.file_uploader(
        "Upload your PDF here",
        type=["pdf"],
        label_visibility="collapsed",
    )
    
    st.caption("Max file size: 25 MB")

    if uploaded and (
        not st.session_state.ingested
        or st.session_state.ingested_filename != uploaded.name
    ):
        with st.status("📥 Ingesting document...", expanded=True) as status:
            st.write(f"**File:** {uploaded.name}")
            st.write("Parsing PDF...")

            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            try:
                from src.app.ingestion.pipeline import run_ingestion_pipeline

                st.write("Chunking & summarizing...")
                vectorstore = run_ingestion_pipeline(
                    tmp_path, persist_dir="chat_chroma_db"
                )
                st.session_state.vectorstore = vectorstore
                st.session_state.ingested = True
                st.session_state.ingested_filename = uploaded.name
                status.update(label="✅ Ready to chat!", state="complete")
            except Exception as e:
                status.update(label="❌ Ingestion failed", state="error")
                st.error(str(e))
            finally:
                os.unlink(tmp_path)

    if st.session_state.ingested:
        st.markdown(
            f'<span class="badge badge-success">● {st.session_state.ingested_filename}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="badge badge-muted">● No document loaded</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ----- Memory viewer ----------
    st.markdown("### 🧠 Memory")

    from src.app.config import USER_MEMORY_PATH, COMPANY_MEMORY_PATH

    user_mem = _read_memory_file(USER_MEMORY_PATH)
    comp_mem = _read_memory_file(COMPANY_MEMORY_PATH)

    with st.expander("👤 User Memory", expanded=bool(user_mem)):
        if user_mem:
            st.markdown(user_mem)
        else:
            st.caption("No user memories yet.")

    with st.expander("🏢 Company Memory", expanded=bool(comp_mem)):
        if comp_mem:
            st.markdown(comp_mem)
        else:
            st.caption("No company memories yet.")

    if st.button("🗑️ Clear Memory", use_container_width=True):
        for path in [USER_MEMORY_PATH, COMPANY_MEMORY_PATH]:
            if os.path.exists(path):
                # Reset to header-only
                with open(path, "w") as f:
                    f.write(
                        f"<!-- {'User' if 'USER' in path else 'Company'} "
                        f"memory — managed by the memory system -->\n"
                    )
        st.toast("🧹 Memory cleared!", icon="✅")
        st.rerun()


# ----- Main chat area ----------
st.markdown("# 💬 Chat with your documents")

if not st.session_state.ingested:
    st.info("👈 Upload a PDF in the sidebar for document Q\u0026A, or just start chatting.")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show citations for assistant messages
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander(
                f"📎 {len(msg['citations'])} citation(s)", expanded=False
            ):
                for c in msg["citations"]:
                    st.markdown(
                        f"**{c.get('source', 'unknown')}** · "
                        f"*{c.get('locator', '')}*\n\n"
                        f"> {c.get('snippet', '')}"
                    )

        # Show memory indicator
        if msg["role"] == "assistant" and msg.get("memory_info"):
            info = msg["memory_info"]
            if info.get("memory_saved"):
                st.caption(
                    f"🧠 Memory updated — "
                    f"{info.get('user_facts_written', 0)} user, "
                    f"{info.get('company_facts_written', 0)} company fact(s)"
                )

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            from src.app.generation.generator import generate_answer
            from src.app.memory import process_memory
            from src.app.routing.router import route_query

            # Step 1: Route the query
            route = route_query(prompt, has_vectorstore=bool(st.session_state.vectorstore))

            # Step 2: Handle each route
            docs = []
            if route == "document_search":
                # Retrieve from documents
                from src.app.retrieval.retriever import get_retriever, retrieve
                retriever = get_retriever(st.session_state.vectorstore)
                docs = retrieve(retriever, prompt)
                result = generate_answer(prompt, docs, mode="rag")
            elif route == "memory_lookup":
                # Answer from memory
                result = generate_answer(prompt, [], mode="memory")
            else:  # route == "general"
                # Conversational answer
                result = generate_answer(prompt, [], mode="general")

            answer = result["answer"]
            citations = result["citations"]

            # Memory
            mem_result = process_memory(prompt, answer)

        # Display answer
        st.markdown(answer)

        # Display citations
        if citations:
            with st.expander(
                f"📎 {len(citations)} citation(s)", expanded=False
            ):
                for c in citations:
                    st.markdown(
                        f"**{c.get('source', 'unknown')}** · "
                        f"*{c.get('locator', '')}*\n\n"
                        f"> {c.get('snippet', '')}"
                    )

        # Memory indicator
        if mem_result.get("memory_saved"):
            st.caption(
                f"🧠 Memory updated — "
                f"{mem_result.get('user_facts_written', 0)} user, "
                f"{mem_result.get('company_facts_written', 0)} company fact(s)"
            )

    # Save to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "memory_info": mem_result,
        }
    )

    # Rerun to refresh sidebar memory viewer
    if mem_result.get("memory_saved"):
        st.rerun()
