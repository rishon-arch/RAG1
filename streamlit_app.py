import os
import io
import re
import math
import uuid
import time
import base64
import warnings
from PIL import Image
import streamlit as st
from dotenv import load_dotenv

# Suppress harmless deprecation and verbose ONNX notices
warnings.filterwarnings("ignore")
import onnxruntime as ort
ort.set_default_logger_severity(3)

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from doc_loader import (
    get_available_documents,
    get_collection_name,
    load_raw_documents,
    split_documents,
    format_citation,
    federated_query,
    SUPPORTED_EXTENSIONS,
    FILE_TYPE_ICONS
)

# Helper for processing uploaded images
def process_uploaded_image(file_obj, max_dim: int = 1024) -> tuple[bytes, str]:
    """Safely inspects, downscales (if needed), and converts an uploaded image to JPEG bytes and base64 string."""
    img = Image.open(file_obj)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img_bytes = buf.getvalue()
    b64_str = base64.b64encode(img_bytes).decode("utf-8")
    return img_bytes, b64_str

def call_groq_resilient(invoke_fn, status_box=None, max_retries: int = 3) -> str:
    """Executes a Groq LLM invocation with exponential backoff on 429 rate limit errors."""
    for attempt in range(max_retries):
        try:
            return invoke_fn()
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str or "quota" in err_str:
                wait_s = 4 + (attempt * 3)
                if status_box:
                    status_box.text(f"⏳ Groq rate limit cooldown ({wait_s}s)...")
                time.sleep(wait_s)
            else:
                raise e
    return "Could not generate answer due to rate limit. Please try again in a few moments."

# Load environment variables
load_dotenv()
load_dotenv(".env.txt")

st.set_page_config(
    page_title="Multi-Document RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# Header
st.title("📚 Multi-Document RAG Assistant")
st.caption("GPU-accelerated federated search across multiple course documents with Groq (Qwen 3.8 27B — Multimodal Vision & Text)")

# Ensure Groq API key is present
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    st.error("Missing GROQ_API_KEY. Please add it to your .env file.")
    st.stop()

DB_DIR = "./chroma_local_db"
os.makedirs("uploaded_docs", exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Load the offline embedding model with AMD GPU DirectML acceleration."""
    return FastEmbedEmbeddings(
        model_name="google/embeddinggemma-300m",
        providers=["DmlExecutionProvider"]
    )

@st.cache_resource(show_spinner=False)
def get_chroma_client():
    """Persistent Chroma client connection."""
    return chromadb.PersistentClient(path=DB_DIR)

embeddings = get_embedding_model()
chroma_client = get_chroma_client()

# Fetch all available documents from project root and uploaded_docs/
available_docs = get_available_documents()
existing_cols = {c.name: c.count() for c in chroma_client.list_collections()}

indexed_docs = []
unindexed_docs = []
doc_map = {}

for d in available_docs:
    col_name = get_collection_name(d["path"])
    cnt = existing_cols.get(col_name, 0)
    d["collection"] = col_name
    d["chunks"] = cnt
    doc_map[d["name"]] = d
    if cnt > 0:
        indexed_docs.append(d)
    else:
        unindexed_docs.append(d)

all_doc_names = [d["name"] for d in available_docs]
indexed_names = [d["name"] for d in indexed_docs]

# Initialize Session State
if "doc_selector" not in st.session_state:
    st.session_state["doc_selector"] = indexed_names.copy()

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
    initial_id = str(uuid.uuid4())[:8]
    st.session_state.chats[initial_id] = {
        "title": "New Chat",
        "messages": []
    }
    st.session_state.current_chat_id = initial_id

def index_single_document(doc_info: dict, status_box, progress_bar):
    """Index a document and persist it to Chroma."""
    doc_name = doc_info["name"]
    doc_path = doc_info["path"]
    col_name = doc_info["collection"]
    
    status_box.text(f"Extracting content from '{doc_name}'...")
    progress_bar.progress(0.05, text=f"Reading '{doc_name}'...")
    
    raw_docs = load_raw_documents(
        doc_path, 
        progress_callback=lambda count, msg: status_box.text(msg)
    )
    
    splits = split_documents(raw_docs, chunk_size=1200, chunk_overlap=200)
    total_chunks = len(splits)
    status_box.text(f"Extracted {len(raw_docs)} items -> {total_chunks} chunks.")
    progress_bar.progress(0.1, text=f"Embedding {total_chunks} chunks locally on GPU...")

    # Clear old collection if re-indexing
    existing = [c.name for c in chroma_client.list_collections()]
    if col_name in existing:
        chroma_client.delete_collection(col_name)

    BATCH_SIZE = 100
    total_batches = math.ceil(total_chunks / BATCH_SIZE)
    new_vs = None

    for idx, i in enumerate(range(0, total_chunks, BATCH_SIZE)):
        batch = splits[i : i + BATCH_SIZE]
        batch_num = idx + 1
        frac = 0.1 + (0.9 * (batch_num / total_batches))
        progress_bar.progress(
            min(frac, 1.0),
            text=f"Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)..."
        )

        if new_vs is None:
            new_vs = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=col_name,
                persist_directory=DB_DIR
            )
        else:
            new_vs.add_documents(documents=batch)

    progress_bar.progress(1.0, text="✅ Ingestion complete!")
    return total_chunks

# ================= TOP-LEVEL INDEXING HANDLER =================
# If an indexing task is pending, run it immediately before rendering any widgets
if st.session_state.get("indexing_doc_name"):
    target_name = st.session_state["indexing_doc_name"]
    target_doc = doc_map.get(target_name)
    if target_doc:
        st.info(f"### ⚙️ Indexing {target_doc['icon']} `{target_name}` on AMD GPU (DirectML)...")
        p_bar = st.progress(0, text="Initializing...")
        s_box = st.empty()
        try:
            chunks = index_single_document(target_doc, s_box, p_bar)
            st.success(f"✅ Successfully indexed **{chunks:,} chunks** for `{target_name}`!")
            # Add to selected docs before widget renders
            current_sel = st.session_state.get("doc_selector", [])
            if target_name not in current_sel:
                st.session_state["doc_selector"] = list(current_sel) + [target_name]
            del st.session_state["indexing_doc_name"]
            time.sleep(1.0)
            st.rerun()
        except Exception as e:
            del st.session_state["indexing_doc_name"]
            st.error(f"Error indexing {target_name}: {e}")
            st.stop()
    else:
        del st.session_state["indexing_doc_name"]

# ================= SIDEBAR =================
with st.sidebar:
    # 1. New Chat button
    if st.button("➕ New Chat", type="primary", use_container_width=True):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.chats[new_id] = {
            "title": "New Chat",
            "messages": []
        }
        st.session_state.current_chat_id = new_id
        st.rerun()

    # 2. Recent Chats
    st.markdown("### 💬 Recent Chats")
    chat_ids = list(st.session_state.chats.keys())
    for cid in reversed(chat_ids):
        cdata = st.session_state.chats[cid]
        col_select, col_del = st.columns([0.82, 0.18])
        is_active = (cid == st.session_state.current_chat_id)
        
        btn_label = f"💬 {cdata['title']}"
        with col_select:
            if st.button(
                btn_label, 
                key=f"chat_{cid}", 
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.current_chat_id = cid
                st.rerun()
        with col_del:
            if st.button("🗑️", key=f"del_{cid}", help="Delete this chat"):
                del st.session_state.chats[cid]
                if st.session_state.current_chat_id == cid:
                    if st.session_state.chats:
                        st.session_state.current_chat_id = list(st.session_state.chats.keys())[-1]
                    else:
                        fresh_id = str(uuid.uuid4())[:8]
                        st.session_state.chats[fresh_id] = {
                            "title": "New Chat",
                            "messages": []
                        }
                        st.session_state.current_chat_id = fresh_id
                st.rerun()

    st.markdown("---")
    
    # 3. Always-Visible Multi-Document Selection
    st.header("📚 Select Documents to Query")
    st.caption("Pick any combination of documents to query simultaneously:")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("✅ Select All", use_container_width=True):
            st.session_state["doc_selector"] = all_doc_names.copy()
            st.rerun()
    with col_btn2:
        if st.button("❌ Clear All", use_container_width=True):
            st.session_state["doc_selector"] = []
            st.rerun()

    # Ensure doc_selector items only contain valid document names
    valid_selector_items = [n for n in st.session_state.get("doc_selector", []) if n in all_doc_names]

    selected_doc_names = st.multiselect(
        "Active Document Scope:",
        options=all_doc_names,
        default=valid_selector_items,
        format_func=lambda n: (
            f"{doc_map[n]['icon']} {n} ({doc_map[n]['chunks']:,} chunks)"
            if doc_map[n]["chunks"] > 0
            else f"{doc_map[n]['icon']} {n} (⚠️ Needs Indexing)"
        ),
        help="Select any combination of documents. Unindexed documents can be indexed right away."
    )
    # Sync back to session state
    st.session_state["doc_selector"] = selected_doc_names

    # Classify selected documents
    ready_target_docs = [doc_map[n] for n in selected_doc_names if doc_map[n]["chunks"] > 0]
    unindexed_selected = [doc_map[n] for n in selected_doc_names if doc_map[n]["chunks"] == 0]

    if ready_target_docs:
        total_ready_chunks = sum(d["chunks"] for d in ready_target_docs)
        st.success(f"Querying **{len(ready_target_docs)} ready doc(s)** ({total_ready_chunks:,} chunks)")
    elif not selected_doc_names:
        st.warning("No documents selected. Choose at least one document above.")

    if unindexed_selected:
        st.warning(f"⚠️ **{len(unindexed_selected)} selected document(s) need indexing**:")
        for un_doc in unindexed_selected:
            if st.button(f"🚀 Index '{un_doc['name']}'", key=f"quick_idx_{un_doc['collection']}", use_container_width=True):
                st.session_state["indexing_doc_name"] = un_doc["name"]
                st.rerun()

    st.markdown("---")

    # 4. Document Library & Uploader
    with st.expander("📁 Document Library & Upload", expanded=True):
        st.caption("Upload new files or manage index collections:")
        
        # Upload new file
        supported_types = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]
        uploaded_file = st.file_uploader(
            "Upload new document",
            type=supported_types,
            help="Supports PDF, Word (.docx), PowerPoint (.pptx), Markdown (.md), Text (.txt), CSV (.csv)"
        )
        auto_index_check = st.checkbox("⚡ Auto-index immediately upon upload", value=True)

        if uploaded_file is not None:
            upload_key = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("last_uploaded_key") != upload_key:
                save_path = os.path.join("uploaded_docs", uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state["last_uploaded_key"] = upload_key
                
                # Auto-trigger indexing if enabled
                if auto_index_check:
                    st.session_state["indexing_doc_name"] = uploaded_file.name
                else:
                    # Just add to selector
                    if uploaded_file.name not in st.session_state.get("doc_selector", []):
                        st.session_state["doc_selector"].append(uploaded_file.name)
                st.success(f"Saved `{uploaded_file.name}`!")
                time.sleep(0.4)
                st.rerun()

        st.markdown("**All Library Documents:**")
        for doc in available_docs:
            dname = doc["name"]
            dicon = doc["icon"]
            dchunks = doc["chunks"]
            
            st.markdown(f"**{dicon} {dname}**")
            col_info, col_btn = st.columns([0.55, 0.45])
            with col_info:
                if dchunks > 0:
                    st.caption(f"✅ {dchunks:,} chunks")
                else:
                    st.caption("⚠️ Not indexed")
            with col_btn:
                btn_label = "🔄 Re-Index" if dchunks > 0 else "🚀 Index"
                if st.button(btn_label, key=f"lib_idx_{doc['collection']}", use_container_width=True):
                    st.session_state["indexing_doc_name"] = doc["name"]
                    st.rerun()
            st.markdown("<hr style='margin:4px 0px 8px 0px;'>", unsafe_allow_html=True)

    st.markdown("---")
    
    # 5. Model Settings
    st.subheader("⚙️ Settings")
    st.write("**LLM:** `Qwen 3.8 27B` (Groq LPU — Multimodal Vision & Text)")
    st.write("**Embeddings:** `EmbeddingGemma 300M (DirectML GPU)`")
    
    top_k = st.slider("Total top chunks to retrieve (k):", min_value=1, max_value=10, value=5, step=1)
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    decompose_multi = st.checkbox(
        "⚡ Multi-Question Auto-Decompose",
        value=True,
        help="Splits multiple numbered questions or MCQs and runs focused federated retrieval for each individually."
    )

# ================= MAIN AREA =================

if not selected_doc_names:
    st.info("👈 Please select at least one document from the sidebar to query.")
    st.stop()

if not ready_target_docs:
    st.warning("⚠️ None of your selected documents are indexed yet. Click **'🚀 Index'** in the sidebar to build their vector database.")
    st.stop()

# Banner displaying active knowledge base scope
active_collections = [d["collection"] for d in ready_target_docs]
if len(ready_target_docs) == 1:
    d = ready_target_docs[0]
    st.caption(f"Active Scope: **{d['icon']} {d['name']}** • {d['chunks']:,} chunks indexed")
else:
    docs_summary = ", ".join([f"{d['name']} ({d['chunks']:,})" for d in ready_target_docs])
    st.caption(f"Active Scope: **🌐 {len(ready_target_docs)} Documents** ({sum(d['chunks'] for d in ready_target_docs):,} total chunks) — *{docs_summary}*")

# Setup Groq LLM
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=temperature,
    max_tokens=1000
)

template = """Answer the question based ONLY on the following context.
Cite the relevant source document and page/slide/row numbers whenever possible.
Format any mathematical expressions using standard LaTeX notation ($...$ or $$...$$).
If the answer is not contained in the context, explicitly say:
"I cannot find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Active chat session
current_chat = st.session_state.chats[st.session_state.current_chat_id]
active_messages = current_chat["messages"]

# Render chat history
for msg in active_messages:
    with st.chat_message(msg["role"]):
        if "images" in msg and msg["images"]:
            for img_bytes in msg["images"]:
                st.image(img_bytes, width=420)
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"📄 View Retrieved Context ({len(msg['sources'])} chunks)"):
                for src in msg["sources"]:
                    dist_info = f" `[dist: {src['distance']:.3f}]`" if "distance" in src else ""
                    st.markdown(f"**{src['source']}{dist_info}:**\n> {src['content']}")

def extract_sub_queries(prompt_text: str) -> list[str]:
    """Detects multiple numbered questions or MCQs and splits them into individual sub-queries."""
    pattern = r'(?:^|\n)(?:(?:\d+[\.\)])|(?:\bQ\d+[:\.]?)|(?:\[\d+\]))\s*'
    splits = [m.start() for m in re.finditer(pattern, prompt_text, re.MULTILINE)]
    if len(splits) >= 2:
        sub_queries = []
        for i in range(len(splits)):
            start = splits[i]
            end = splits[i+1] if i + 1 < len(splits) else len(prompt_text)
            q = prompt_text[start:end].strip()
            if q:
                sub_queries.append(q)
        return sub_queries
    return [prompt_text.strip()]

def format_hits_for_context(hits: list[dict]) -> tuple[str, list[dict]]:
    """Format federated hits into LLM context string and UI source metadata."""
    formatted_chunks = []
    sources = []
    for h in hits:
        meta = h.get("metadata", {})
        cit = format_citation(meta)
        src_file = os.path.basename(meta.get("source", h.get("collection", "Document")))
        
        # Context string for LLM
        formatted_chunks.append(f"[{src_file} - {cit}]:\n{h['content']}")
        
        # Source object for UI drawer
        sources.append({
            "source": f"{src_file} - {cit}",
            "distance": h.get("distance", 0.0),
            "content": h["content"].strip()
        })
    return "\n\n".join(formatted_chunks), sources

# Chat Input (accepts text, images, or both)
chat_submission = st.chat_input(
    "Ask a question or attach an image/diagram across your selected documents...",
    accept_file=True,
    file_type=["png", "jpg", "jpeg", "webp"]
)

if chat_submission:
    raw_text = chat_submission.text.strip() if hasattr(chat_submission, "text") else (chat_submission.strip() if isinstance(chat_submission, str) else "")
    uploaded_files = chat_submission.files if hasattr(chat_submission, "files") else []
    
    if not raw_text and not uploaded_files:
        st.stop()
        
    uploaded_img_bytes = []
    uploaded_img_b64 = []
    for f in uploaded_files:
        try:
            b_bytes, b_b64 = process_uploaded_image(f)
            uploaded_img_bytes.append(b_bytes)
            uploaded_img_b64.append(b_b64)
        except Exception as img_err:
            st.warning(f"Could not process image '{getattr(f, 'name', 'file')}': {img_err}")
            
    is_multimodal = len(uploaded_img_b64) > 0
    search_query = raw_text
    display_user_query = raw_text
    
    # If the user only uploaded an image without typing text:
    if is_multimodal and not raw_text:
        with st.chat_message("user"):
            for b in uploaded_img_bytes:
                st.image(b, width=420)
            status_extract = st.empty()
            status_extract.caption("🔍 Reading visual contents and diagram from image...")
            
            extract_prompt = (
                "Transcribe and summarize the key technical question, problem, equations, or diagram in this image "
                "in 1-2 concise search sentences so relevant course materials can be retrieved from the vector database."
            )
            extract_msg = HumanMessage(content=[
                {"type": "text", "text": extract_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{uploaded_img_b64[0]}"}}
            ])
            extracted_search_query = call_groq_resilient(lambda: llm.invoke([extract_msg]).content.strip(), status_box=status_extract)
            status_extract.empty()
            
        display_user_query = "📷 *Analyze and solve the problem/diagram shown in the image.*"
        search_query = extracted_search_query
    else:
        # Standard user message render
        with st.chat_message("user"):
            if uploaded_img_bytes:
                for b in uploaded_img_bytes:
                    st.image(b, width=420)
            st.markdown(display_user_query)

    # Rename "New Chat" to the first user question
    if not active_messages or current_chat["title"] == "New Chat":
        if raw_text:
            clean_title = raw_text.strip()
        elif is_multimodal:
            clean_title = "📷 Image Query"
        else:
            clean_title = "New Chat"
        current_chat["title"] = (clean_title[:28] + "...") if len(clean_title) > 28 else clean_title

    # Append user message to history
    user_msg_entry = {
        "role": "user",
        "content": display_user_query
    }
    if uploaded_img_bytes:
        user_msg_entry["images"] = uploaded_img_bytes
    active_messages.append(user_msg_entry)

    sub_queries = extract_sub_queries(search_query) if (decompose_multi and not is_multimodal) else [search_query]

    # Generate Assistant Response
    if len(sub_queries) > 1:
        with st.chat_message("assistant"):
            status_box = st.empty()
            status_box.info(f"⚡ Detected **{len(sub_queries)} questions**. Running federated retrieval across {len(active_collections)} collections...")

            all_answers = []
            all_sources = []
            chain = prompt | llm | StrOutputParser()

            sub_k = min(top_k, 4)
            for idx, sub_q in enumerate(sub_queries):
                status_box.text(f"Processing Question {idx + 1}/{len(sub_queries)} (searching top {sub_k} chunks)...")

                hits = federated_query(
                    client=chroma_client,
                    collection_names=active_collections,
                    query_text=sub_q,
                    embeddings=embeddings,
                    k_per_col=sub_k,
                    total_k=sub_k
                )

                context_str, sources = format_hits_for_context(hits)
                for s in sources:
                    s["source"] = f"Q{idx + 1}: {s['source']}"
                    all_sources.append(s)

                ans = call_groq_resilient(
                    lambda: chain.invoke({
                        "context": context_str,
                        "question": sub_q
                    }),
                    status_box=status_box
                )

                all_answers.append(f"### Question {idx + 1}\n{ans}")
                time.sleep(0.3)

            status_box.empty()
            response = "\n\n---\n\n".join(all_answers)
            st.markdown(response)

            if all_sources:
                with st.expander(f"📄 View Retrieved Context ({len(all_sources)} chunks across {len(sub_queries)} questions)"):
                    for src in all_sources:
                        dist_info = f" `[dist: {src['distance']:.3f}]`" if "distance" in src else ""
                        st.markdown(f"**{src['source']}{dist_info}:**\n> {src['content']}")

            active_messages.append({
                "role": "assistant",
                "content": response,
                "sources": all_sources
            })
            st.rerun()

    else:
        # Single Query Flow (Text-only or Multimodal)
        with st.chat_message("assistant"):
            spin_label = "Analyzing image and searching document collection(s)..." if is_multimodal else f"Searching across {len(active_collections)} document collection(s)..."
            with st.spinner(spin_label):
                hits = federated_query(
                    client=chroma_client,
                    collection_names=active_collections,
                    query_text=search_query,
                    embeddings=embeddings,
                    k_per_col=top_k,
                    total_k=top_k
                )

                context_str, sources = format_hits_for_context(hits)
                
                if is_multimodal:
                    multimodal_content = [
                        {
                            "type": "text",
                            "text": (
                                f"{template.split('Context:')[0].strip()}\n\n"
                                f"Context from course materials:\n{context_str}\n\n"
                                f"User Question / Prompt:\n{display_user_query if raw_text else search_query}\n\n"
                                f"Answer the question based on the provided context and the attached image. "
                                f"Cite relevant source documents and slide/page numbers wherever possible, "
                                f"and format all mathematical expressions using LaTeX notation ($...$ or $$...$$).\n\n"
                                f"Answer:"
                            )
                        }
                    ]
                    for b64 in uploaded_img_b64:
                        multimodal_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        })
                    
                    human_msg = HumanMessage(content=multimodal_content)
                    response = call_groq_resilient(lambda: llm.invoke([human_msg]).content)
                else:
                    chain = prompt | llm | StrOutputParser()
                    response = call_groq_resilient(lambda: chain.invoke({
                        "context": context_str,
                        "question": search_query
                    }))

                st.markdown(response)

                if sources:
                    with st.expander(f"📄 View Retrieved Context ({len(sources)} chunks)"):
                        for src in sources:
                            dist_info = f" `[dist: {src['distance']:.3f}]`" if "distance" in src else ""
                            st.markdown(f"**{src['source']}{dist_info}:**\n> {src['content']}")

            active_messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })
            st.rerun()

