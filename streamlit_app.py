import os
import re
import math
import uuid
import warnings
import streamlit as st
from dotenv import load_dotenv

# Suppress harmless deprecation and verbose ONNX notices
warnings.filterwarnings("ignore")
import onnxruntime as ort
ort.set_default_logger_severity(3)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment keys
load_dotenv()
load_dotenv(".env.txt")

st.set_page_config(
    page_title="RAG Course Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom header
st.title("📚 Course Material Assistant")
st.caption("Interactive Q&A powered by LangChain, ChromaDB, FastEmbed (DirectML GPU) & Groq (Qwen 3.8 27B)")

# Ensure Groq API key is present for generation
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    st.error("Missing GROQ_API_KEY. Please add it to your .env file.")
    st.stop()

DB_DIR = "./chroma_local_db"
PDF_PATH = "nptel.pdf"

@st.cache_resource(show_spinner=False)
def get_rag_components():
    """Load the persistent vector store and offline embedding model with AMD GPU DirectML."""
    embeddings = FastEmbedEmbeddings(
        model_name="google/embeddinggemma-300m",
        providers=["DmlExecutionProvider"]
    )
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        try:
            vectorstore = Chroma(
                persist_directory=DB_DIR,
                embedding_function=embeddings
            )
            if vectorstore._collection.count() > 0:
                return vectorstore, embeddings
        except Exception:
            pass
    return None, embeddings

vectorstore, embeddings = get_rag_components()

# Manage Multi-Chat Sessions (like Gemini / ChatGPT)
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
    initial_id = str(uuid.uuid4())[:8]
    st.session_state.chats[initial_id] = {
        "title": "New Chat",
        "messages": []
    }
    st.session_state.current_chat_id = initial_id

# Sidebar
with st.sidebar:
    # Prominent Gemini-style "+ New Chat" button at top
    if st.button("➕ New Chat", type="primary", use_container_width=True):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.chats[new_id] = {
            "title": "New Chat",
            "messages": []
        }
        st.session_state.current_chat_id = new_id
        st.rerun()

    # Chat History list (click to switch, trash to delete)
    st.markdown("### 💬 Recent Chats")
    chat_ids = list(st.session_state.chats.keys())
    for cid in reversed(chat_ids):
        cdata = st.session_state.chats[cid]
        col_select, col_del = st.columns([0.82, 0.18])
        is_active = (cid == st.session_state.current_chat_id)
        
        # Display title with active marker
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
                        st.session_state.chats[fresh_id] = {"title": "New Chat", "messages": []}
                        st.session_state.current_chat_id = fresh_id
                st.rerun()

    st.markdown("---")
    st.header("⚙️ Configuration")
    st.write(f"**Document:** `{PDF_PATH}`")
    
    if vectorstore and vectorstore._collection.count() > 0:
        chunk_count = vectorstore._collection.count()
        st.success(f"Vector DB Active ({chunk_count} chunks)")
    else:
        st.warning("Vector DB not indexed yet.")

    st.markdown("---")
    st.subheader("Model Settings")
    st.write("**LLM:** `Qwen 3.8 27B` (via Groq)")
    st.write("**Embeddings:** `EmbeddingGemma 300M (AMD GPU DirectML)`")
    
    top_k = st.slider("Top chunks to retrieve (k):", min_value=1, max_value=6, value=3, step=1)
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    decompose_multi = st.checkbox(
        "⚡ Multi-Question Auto-Decompose",
        value=True,
        help="Automatically detects multiple questions or MCQs in a prompt and runs focused retrieval (k=3) for each individually, guaranteeing accurate answers without hitting API limits."
    )

# One-time In-Browser Indexing Flow (if DB is ever deleted)
if vectorstore is None or vectorstore._collection.count() == 0:
    st.info(f"The vector database for `{PDF_PATH}` has not been created yet.")
    st.write("Click the button below to extract and embed the document locally on your **AMD GPU (DirectML)** using **EmbeddingGemma**. This is a **one-time process**; all subsequent visits will load instantly from disk.")
    
    if not os.path.exists(PDF_PATH):
        st.error(f"File `{PDF_PATH}` was not found in the project folder.")
        st.stop()

    if st.button("🚀 Index Document Now (One-Time)", type="primary", use_container_width=False):
        progress_bar = st.progress(0, text="Reading pages and splitting text...")
        status_box = st.empty()
        
        try:
            # 1. Lazy load and split
            loader = PyPDFLoader(PDF_PATH)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200
            )
            splits = []
            page_count = 0
            for page in loader.lazy_load():
                splits.extend(text_splitter.split_documents([page]))
                page_count += 1
                if page_count % 150 == 0:
                    status_box.text(f"Processed {page_count} pages...")

            total_chunks = len(splits)
            status_box.text(f"Total pages processed: {page_count}. Total chunks: {total_chunks}.")
            progress_bar.progress(0.1, text=f"Extracted {total_chunks} chunks. Embedding locally with Gemma on GPU...")

            # 2. Local batch embedding into Chroma
            BATCH_SIZE = 100
            total_batches = math.ceil(total_chunks / BATCH_SIZE)
            new_vs = None

            for idx, i in enumerate(range(0, total_chunks, BATCH_SIZE)):
                batch = splits[i : i + BATCH_SIZE]
                batch_num = idx + 1
                frac = 0.1 + (0.9 * (batch_num / total_batches))
                progress_bar.progress(min(frac, 1.0), text=f"Embedding batch {batch_num} of {total_batches} ({len(batch)} chunks)...")

                if new_vs is None:
                    new_vs = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=DB_DIR
                    )
                else:
                    new_vs.add_documents(documents=batch)

            progress_bar.progress(1.0, text="✅ Ingestion complete! Reloading assistant...")
            st.success(f"Successfully indexed {total_chunks} chunks into `{DB_DIR}`!")
            
            # Clear resource cache and refresh to load the ready RAG chat UI
            st.cache_resource.clear()
            st.rerun()

        except Exception as e:
            st.error(f"Error during indexing: {e}")
            st.stop()
    else:
        st.stop()

# Initialize ChatGroq LLM
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=temperature
)

retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

# RAG Prompt Template
template = """Answer the question based ONLY on the following context.
Cite the relevant page numbers whenever possible.
Format any mathematical expressions using standard LaTeX notation ($...$ or $$...$$).
If the answer is not contained in the context, explicitly say:
"I cannot find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Active chat session
current_chat = st.session_state.chats[st.session_state.current_chat_id]
active_messages = current_chat["messages"]

# Display prior chat messages for the current chat session
for msg in active_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 View Retrieved Page Context"):
                for src in msg["sources"]:
                    st.markdown(f"**Page {src['page']}:**\n> {src['content']}")

def extract_sub_queries(prompt_text: str) -> list[str]:
    """
    Detects if the user entered multiple questions (e.g. 1., 2., Q1, [1])
    and splits them into individual sub-queries.
    """
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

# User query input
if user_query := st.chat_input("Ask any question from the course material..."):
    # Automatically rename "New Chat" to the first question (like Gemini / ChatGPT)
    if not active_messages or current_chat["title"] == "New Chat":
        clean_title = user_query.strip()
        current_chat["title"] = (clean_title[:28] + "...") if len(clean_title) > 28 else clean_title

    # Add user message
    active_messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    sub_queries = extract_sub_queries(user_query) if decompose_multi else [user_query]

    # Generate assistant response
    if len(sub_queries) > 1:
        with st.chat_message("assistant"):
            status_box = st.empty()
            status_box.info(f"⚡ Detected **{len(sub_queries)} questions**. Running focused retrieval & answer generation for each...")
            
            all_answers = []
            all_sources = []
            chain = prompt | llm | StrOutputParser()

            sub_k = min(top_k, 3)
            for idx, sub_q in enumerate(sub_queries):
                status_box.text(f"Processing Question {idx + 1} of {len(sub_queries)} (using top {sub_k} focused chunks)...")
                
                # Dedicated retrieval for each question with focused context (k=3)
                docs = vectorstore.similarity_search(sub_q, k=sub_k)
                
                formatted_chunks = []
                for d in docs:
                    page_num = d.metadata.get("page", "Unknown")
                    formatted_chunks.append(f"[Page {page_num}]:\n{d.page_content}")
                    all_sources.append({
                        "page": f"Q{idx + 1} (Page {page_num})",
                        "content": d.page_content.strip()
                    })
                
                context_str = "\n\n".join(formatted_chunks)
                
                # Resilient Groq call with auto-retry on TPM/RPM rate limits
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        ans = chain.invoke({
                            "context": context_str,
                            "question": sub_q
                        })
                        break
                    except Exception as e:
                        err_str = str(e).lower()
                        if "429" in err_str or "rate" in err_str or "quota" in err_str:
                            wait_s = 4 + (attempt * 3)
                            status_box.text(f"⏳ Groq token limit reached. Cooling down {wait_s}s before retrying Question {idx + 1}...")
                            time.sleep(wait_s)
                        else:
                            raise e
                else:
                    ans = "Could not generate answer due to repeated rate limit."

                all_answers.append(f"### Question {idx + 1}\n{ans}")
                
                # Small 0.3s pause between calls to smoothly stay under Groq TPM limits
                time.sleep(0.3)

            status_box.empty()
            response = "\n\n---\n\n".join(all_answers)
            st.markdown(response)

            if all_sources:
                with st.expander(f"📄 View Retrieved Page Context ({len(all_sources)} chunks across {len(sub_queries)} questions)"):
                    for src in all_sources:
                        st.markdown(f"**{src['page']}:**\n> {src['content']}")
            
            sources = all_sources
    else:
        # Standard single query flow
        with st.chat_message("assistant"):
            with st.spinner("Searching document context and generating answer..."):
                docs = retriever.invoke(user_query)
                formatted_chunks = []
                sources = []
                for d in docs:
                    page_num = d.metadata.get("page", "Unknown")
                    formatted_chunks.append(f"[Page {page_num}]:\n{d.page_content}")
                    sources.append({
                        "page": page_num,
                        "content": d.page_content.strip()
                    })
                
                context_str = "\n\n".join(formatted_chunks)
                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({
                    "context": context_str,
                    "question": user_query
                })

                st.markdown(response)

                if sources:
                    with st.expander("📄 View Retrieved Page Context"):
                        for src in sources:
                            st.markdown(f"**Page {src['page']}:**\n> {src['content']}")

    # Save to chat history for this session
    active_messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })
    st.rerun()
