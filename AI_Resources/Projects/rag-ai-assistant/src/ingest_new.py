# === src/ingest_new.py ===
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def load_and_chunk_all_pdfs(data_dir, chunk_size=500, chunk_overlap=50):
    all_chunks = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Directory not found: {data_dir}")

    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_dir, filename)
            print(f"📄 Loading: {filename}")
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()

                for doc in documents:
                    doc.metadata["source"] = filename  # Important!

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ".", " ", ""]
                )
                chunks = splitter.split_documents(documents)
                print(f"✅ Chunked into {len(chunks)} pieces from {filename}")
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"⚠️ Failed to load {filename}: {e}")

    print(f"\n📚 Total chunks from all PDFs: {len(all_chunks)}")
    return all_chunks


def embed_and_store(chunks, persist_dir):
    if not chunks:
        print("❌ No chunks to embed. Exiting.")
        return

    print("🔍 Generating embeddings using MiniLM...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("💾 Saving vector index...")
    vectorstore.save_local(persist_dir)
    print(f"✅ Vector store saved at: {persist_dir}")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(base_dir, "..", "data"))
    vector_dir = os.path.abspath(os.path.join(base_dir, "..", "vectorstore"))

    print(f"📂 Using data directory: {data_dir}")
    print(f"📦 Saving vectorstore to: {vector_dir}\n")

    chunks = load_and_chunk_all_pdfs(data_dir)
    embed_and_store(chunks, vector_dir)
