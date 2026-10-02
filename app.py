import os
import time
import math
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
load_dotenv(".env.txt")
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Missing GROQ_API_KEY. Check your .env file.")

DB_DIR = "./chroma_local_db"
PDF_PATH = "nptel.pdf"

# Initialize offline EmbeddingGemma model via FastEmbed with AMD GPU DirectML acceleration
embeddings = FastEmbedEmbeddings(
    model_name="google/embeddinggemma-300m",
    providers=["DmlExecutionProvider"]
)

# Check if DB already exists and has documents
vectorstore = None
if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
    try:
        vs = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )
        count = vs._collection.count()
        if count > 0:
            print(f"Loading existing vector database ({count} chunks) from '{DB_DIR}'...")
            vectorstore = vs
        else:
            print(f"Existing vector database at '{DB_DIR}' is empty. Re-indexing...")
    except Exception:
        print(f"Error reading existing vector store. Rebuilding from '{PDF_PATH}'...")

if vectorstore is None:
    print(f"Indexing document: '{PDF_PATH}' using offline EmbeddingGemma...")
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"'{PDF_PATH}' not found.")

    # 1. Stream pages lazily
    loader = PyPDFLoader(PDF_PATH)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    print("Extracting pages and creating chunks...")
    splits = []
    page_count = 0
    for page in loader.lazy_load():
        splits.extend(text_splitter.split_documents([page]))
        page_count += 1
        if page_count % 100 == 0:
            print(f"  Processed {page_count} pages ({len(splits)} chunks created)...")

    total_chunks = len(splits)
    print(f"Finished extraction: {total_chunks} total chunks to embed.")

    # 2. Local ingestion into ChromaDB (FastEmbed runs offline on CPU - no API quota or 429 rate limits)
    BATCH_SIZE = 100
    total_batches = math.ceil(total_chunks / BATCH_SIZE)

    for idx, i in enumerate(range(0, total_chunks, BATCH_SIZE)):
        batch = splits[i : i + BATCH_SIZE]
        batch_num = idx + 1
        print(f"Embedding batch {batch_num} of {total_batches} ({len(batch)} chunks)...")

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=DB_DIR
            )
        else:
            vectorstore.add_documents(documents=batch)

    print("All chunks indexed locally and persisted to disk!")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Setup Groq LLM (Qwen 3.8 27B)
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.2
)

template = """Answer the question based ONLY on the following context.
Cite the relevant page numbers whenever possible.
If the answer is not contained in the context, explicitly say:
"I cannot find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    formatted = []
    for d in docs:
        page_num = d.metadata.get("page", "Unknown")
        formatted.append(f"[Page {page_num}]:\n{d.page_content}")
    return "\n\n".join(formatted)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(f"\nRAG System Ready! Ask questions about your document ('{PDF_PATH}'):\n")
while True:
    query = input("Question: ").strip()
    if query.lower() in ("exit", "quit", "q"):
        break
    if not query:
        continue

    response = rag_chain.invoke(query)
    print(f"\nQwen (Groq):\n{response}\n" + "-" * 50)