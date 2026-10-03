import os
import sys
import math
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
import onnxruntime as ort
ort.set_default_logger_severity(3)

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from doc_loader import (
    get_available_documents,
    get_collection_name,
    load_raw_documents,
    split_documents,
    format_citation,
    federated_query
)

load_dotenv()
load_dotenv(".env.txt")
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Missing GROQ_API_KEY. Check your .env file.")

DB_DIR = "./chroma_local_db"
os.makedirs(DB_DIR, exist_ok=True)

def main():
    available = get_available_documents()
    if not available:
        print("No supported documents found in the current directory.")
        sys.exit(1)

    client = chromadb.PersistentClient(path=DB_DIR)
    existing_cols = {c.name: c.count() for c in client.list_collections()}

    indexed_docs = []
    unindexed_docs = []
    for d in available:
        c_name = get_collection_name(d["path"])
        cnt = existing_cols.get(c_name, 0)
        d["collection"] = c_name
        d["chunks"] = cnt
        if cnt > 0:
            indexed_docs.append(d)
        else:
            unindexed_docs.append(d)

    # 1. Document Selection (CLI arguments or interactive selection)
    selected_docs = []
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        if args[0].lower() in ("all", "*"):
            selected_docs = indexed_docs.copy()
        else:
            for arg in args:
                for doc in available:
                    if doc["name"].lower() == arg.lower() or doc["path"].lower() == arg.lower():
                        selected_docs.append(doc)
                        break

    if not selected_docs:
        print("\n" + "=" * 65)
        print("Multi-Document Knowledge Base:")
        print("=" * 65)
        total_indexed_chunks = sum(d["chunks"] for d in indexed_docs)
        if indexed_docs:
            print(f" [0] 🌐 ALL Indexed Documents ({len(indexed_docs)} docs, {total_indexed_chunks:,} chunks)")
        
        for idx, doc in enumerate(available, 1):
            status = f"Indexed: {doc['chunks']:,} chunks" if doc['chunks'] > 0 else "Not indexed"
            print(f" [{idx}] {doc['name']} ({doc['size_mb']} MB) [{status}]")
        print("=" * 65)
        
        choice = input(f"Select document numbers (e.g. '0' for all, or '1,2') [default: 0]: ").strip()
        if not choice or choice == "0":
            selected_docs = indexed_docs.copy() if indexed_docs else [available[0]]
        else:
            for part in choice.split(","):
                part = part.strip()
                try:
                    val = int(part)
                    if 1 <= val <= len(available):
                        selected_docs.append(available[val - 1])
                except ValueError:
                    pass
            if not selected_docs:
                selected_docs = [available[0]]

    # 2. Check for any unindexed documents in selection
    embeddings = FastEmbedEmbeddings(
        model_name="google/embeddinggemma-300m",
        providers=["DmlExecutionProvider"]
    )

    for doc in selected_docs:
        col_name = doc["collection"]
        cnt = existing_cols.get(col_name, 0)
        if cnt == 0:
            print(f"\nDocument '{doc['name']}' is not indexed yet. Indexing now on GPU...")
            raw_docs = load_raw_documents(doc["path"])
            splits = split_documents(raw_docs, chunk_size=1200, chunk_overlap=200)
            total_chunks = len(splits)
            print(f"Extracted {len(raw_docs)} items -> {total_chunks} total chunks to embed.")

            BATCH_SIZE = 100
            total_batches = math.ceil(total_chunks / BATCH_SIZE)
            new_vs = None
            for idx, i in enumerate(range(0, total_chunks, BATCH_SIZE)):
                batch = splits[i : i + BATCH_SIZE]
                batch_num = idx + 1
                print(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
                if new_vs is None:
                    new_vs = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        collection_name=col_name,
                        persist_directory=DB_DIR
                    )
                else:
                    new_vs.add_documents(documents=batch)
            print(f"Indexed {total_chunks} chunks into collection '{col_name}'.")
            doc["chunks"] = total_chunks

    active_collections = [d["collection"] for d in selected_docs]
    doc_names_str = ", ".join([d["name"] for d in selected_docs])
    print(f"\nActive Scope: {len(selected_docs)} document(s): [{doc_names_str}]")
    print(f"Chroma Collections: {active_collections}")

    # 3. Setup Groq LLM
    llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0.2
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

    def format_hits(hits):
        formatted = []
        for h in hits:
            meta = h.get("metadata", {})
            cit = format_citation(meta)
            src_file = os.path.basename(meta.get("source", h.get("collection", "Document")))
            formatted.append(f"[{src_file} - {cit}]:\n{h['content']}")
        return "\n\n".join(formatted)

    chain = prompt | llm | StrOutputParser()

    print(f"\nRAG System Ready! Ask questions across your selected document(s) (type 'exit' or 'q' to quit):\n")
    while True:
        query = input("Question: ").strip()
        if query.lower() in ("exit", "quit", "q"):
            break
        if not query:
            continue

        hits = federated_query(
            client=client,
            collection_names=active_collections,
            query_text=query,
            embeddings=embeddings,
            k_per_col=3,
            total_k=5
        )

        context_str = format_hits(hits)
        response = chain.invoke({"context": context_str, "question": query})

        print(f"\nQwen (Groq):\n{response}\n")
        print("Sources Retrieved:")
        for h in hits:
            meta = h.get("metadata", {})
            cit = format_citation(meta)
            src_file = os.path.basename(meta.get("source", h.get("collection", "Document")))
            print(f" - [{src_file} - {cit}] (dist: {h.get('distance', 0.0):.3f})")
        print("-" * 65)

if __name__ == "__main__":
    main()