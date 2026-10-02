import os
import time
import math
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
load_dotenv(".env.txt")

DB_DIR = "./chroma_local_db"
PDF_PATH = "nptel.pdf"

print("=" * 60)
print("BENCHMARK: DirectML (AMD Radeon GPU) Indexing with EmbeddingGemma")
print("=" * 60)

# Check providers
import onnxruntime as ort
print("ONNX Runtime Available Providers:", ort.get_available_providers())

t_start_total = time.time()

# 1. Initialize FastEmbed with DmlExecutionProvider
t0 = time.time()
print("\n[1/3] Initializing FastEmbed with 'google/embeddinggemma-300m' on DmlExecutionProvider (AMD GPU)...")
embeddings = FastEmbedEmbeddings(
    model_name="google/embeddinggemma-300m",
    providers=["DmlExecutionProvider"]
)
print(f"Model initialized in {time.time() - t0:.2f}s")

# 2. Extract and split pages
t0 = time.time()
print(f"\n[2/3] Extracting pages from '{PDF_PATH}' (1,115 pages)...")
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
    if page_count % 200 == 0:
        print(f"  Processed {page_count} pages ({len(splits)} chunks created)...")

total_chunks = len(splits)
extraction_time = time.time() - t0
print(f"Extraction complete in {extraction_time:.2f}s: {page_count} pages -> {total_chunks} total chunks.")

# 3. Batch GPU Ingestion into Chroma
t0_embed = time.time()
BATCH_SIZE = 100
total_batches = math.ceil(total_chunks / BATCH_SIZE)
print(f"\n[3/3] Embedding {total_chunks} chunks on GPU via DirectML (in {total_batches} batches of {BATCH_SIZE})...")

vectorstore = None
for idx, i in enumerate(range(0, total_chunks, BATCH_SIZE)):
    t_batch_start = time.time()
    batch = splits[i : i + BATCH_SIZE]
    batch_num = idx + 1

    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            documents=batch,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
    else:
        vectorstore.add_documents(documents=batch)

    batch_time = time.time() - t_batch_start
    rate = len(batch) / batch_time
    print(f"  Batch {batch_num:2d}/{total_batches} ({len(batch):3d} chunks) embedded in {batch_time:5.2f}s ({rate:5.1f} chunks/sec)")

embed_time = time.time() - t0_embed
total_time = time.time() - t_start_total

print("\n" + "=" * 60)
print("BENCHMARK RESULTS (AMD Radeon 860M GPU / DirectML)")
print("=" * 60)
print(f"Total Chunks:           {total_chunks}")
print(f"PDF Extraction Time:    {extraction_time:.2f}s")
print(f"GPU Embedding Time:     {embed_time:.2f}s ({embed_time/60:.2f} min)")
print(f"Average GPU Speed:      {total_chunks / embed_time:.1f} chunks/sec")
print(f"Total End-to-End Time:  {total_time:.2f}s ({total_time/60:.2f} min)")
print(f"Saved DB Chunk Count:   {vectorstore._collection.count()}")
print("=" * 60)
