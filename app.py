import streamlit as st
import requests
import pandas as pd
from urllib.parse import quote

# Page config (use full width)
st.set_page_config(page_title="researchSoup", layout="wide", page_icon="🥣")

# Custom CSS to remove container padding
st.markdown(
    "<style>"
    ".stApp {max-width: 100% !important; padding: 0 1rem;}"
    "footer {padding-top: 2rem;}"
    "</style>",
    unsafe_allow_html=True
)

# Header
st.markdown(
    "<h1 style='text-align:center; color:#f5a623;'>🥣 researchSoup</h1>"
    "<p style='text-align:center; color:#e0c097;'>Serve your queries and sip on responses</p>",
    unsafe_allow_html=True
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_table" not in st.session_state:
    st.session_state.docs_table = []
if "pdf_paths" not in st.session_state:
    st.session_state.pdf_paths = []
if "new_prompt" not in st.session_state:
    st.session_state.new_prompt = ''

# Tabs: Chat & Ingest
tab_chat, tab_ingest = st.tabs(["🍽️ Chat", "📚 Ingest Documents"])

with tab_chat:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'], unsafe_allow_html=True)

    # Documents expander
    if st.session_state.docs_table:
        with st.expander("📄 Retrieved Documents", expanded=True):
            st.dataframe(pd.DataFrame(st.session_state.docs_table), use_container_width=True)
            if st.session_state.pdf_paths:
                st.markdown("**Downloadable PDFs:**")
                cols = st.columns(min(3, len(st.session_state.pdf_paths)))
                for i, path in enumerate(st.session_state.pdf_paths):
                    label = f"Document #{i+1:02d}"
                    url = f"http://localhost:8001/download?file_path={quote(path)}"
                    with cols[i % 3]:
                        try:
                            data = requests.get(url, timeout=10).content
                            st.download_button(
                                label=label,
                                data=data,
                                file_name=path.rsplit('/',1)[-1],
                                mime="application/pdf",
                                key=f"dl_{i}"
                            )
                        except:
                            st.markdown(f"[{label}]({url})")

    # Chat Input fixed at bottom: wide columns
    input_col, send_col = st.columns([10, 1])
    st.session_state.new_prompt = input_col.text_input(
        "",
        value=st.session_state.new_prompt,
        placeholder="What's cooking in your mind?",
        key="chat_input",
        label_visibility="collapsed",
        max_chars=500,
        help="Type your question here"
    )
    send = send_col.button("Send", use_container_width=True)

    if send and st.session_state.new_prompt.strip():
        prompt = st.session_state.new_prompt.strip()
        st.session_state.messages.append({'role':'user','content':prompt})
        try:
            r = requests.post(
                "http://localhost:8002/chat",
                json={'query': prompt},
                timeout=30
            )
            r.raise_for_status()
            resp = r.json().get('response', {})
            answer = resp.get('response', 'No answer.')
            st.session_state.messages.append({'role':'assistant','content':f"**Answer:**\n{answer}"})

            # Process documents
            raw = resp.get('documents', {}).get('results', [])
            docs = []
            if isinstance(raw, list) and raw:
                if isinstance(raw[0], list) and len(raw[0]) > 1:
                    docs = raw[0][1]
                elif isinstance(raw[0], dict):
                    docs = raw

            table, pdfs, seen = [], [], set()
            for idx, d in enumerate(docs, 1):
                text = d.get('payload', {}).get('text', '')
                preview = text[:200] + ('...' if len(text) > 200 else '')
                table.append({'Doc #': idx, 'Score': round(d.get('score', 0), 3), 'Preview': preview})
                path = d.get('payload', {}).get('pdf_path')
                if path and path not in seen:
                    pdfs.append(path)
                    seen.add(path)

            st.session_state.docs_table = table
            st.session_state.pdf_paths = pdfs
        except Exception as e:
            st.session_state.messages.append({'role':'assistant','content':f"⚠️ {e}"})
        # Clear input and rerun
        st.session_state.new_prompt = ''
        st.rerun()

with tab_ingest:
    st.markdown("### 📥 Upload Research Documents")
    st.markdown("Add new documents to your research collection")
    with st.form("ingest_form", clear_on_submit=True):
        researcher = st.text_input("Researcher Name")
        collection = st.text_input("Collection Name", value="AI_store")
        findings = st.text_area("Research Findings")
        model = st.selectbox(
            "Embedding Model (Optional)",
            ["Default", "BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"]
        )
        files = st.file_uploader("Upload PDFs", type='pdf', accept_multiple_files=True)
        if st.form_submit_button("🚀 Ingest Documents"):
            if not (researcher and findings and files):
                st.error("Fill in all fields and upload at least one PDF.")
            else:
                payload = {'researcher': researcher, 'findings': findings, 'collection_name': collection}
                if model != "Default":
                    payload['embedding_model'] = model
                files_payload = [('pdfs', (f.name, f.getvalue(), 'application/pdf')) for f in files]
                with st.spinner("Ingesting documents...⏳"):
                    try:
                        r = requests.post(
                            "http://localhost:8001/ingest_documents",
                            data=payload,
                            files=files_payload,
                            timeout=60
                        )
                        r.raise_for_status()
                        res = r.json()
                        ins, sk = res.get('inserted_docs', []), res.get('skipped_docs', [])
                        st.success(f"Processed: {len(ins)} inserted, {len(sk)} skipped.")
                        def show_list(t, docs):
                            if not docs: return ""
                            items = "".join(
                                f"<li>{doc.get('text','')[:100]}...</li>" for doc in docs
                            )
                            return f"<details><summary>{t} ({len(docs)})</summary><ul>{items}</ul></details>"
                        st.markdown(show_list("Inserted", ins), unsafe_allow_html=True)
                        st.markdown(show_list("Skipped", sk), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ {e}")

# Footer
st.markdown(
    "<footer style='text-align:center; color:#888; font-size:0.8rem;'>"
    "researchSoup 🍲 — brewed with 🤖 and ☕</footer>", unsafe_allow_html=True
)