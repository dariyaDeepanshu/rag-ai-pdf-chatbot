import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA


# 🔹 Load FAISS vectorstore
def load_vectorstore(vectorstore_path):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        folder_path=vectorstore_path,
        embeddings=embeddings,
        index_name="index",
        allow_dangerous_deserialization=True
    )


# 🔹 Load GGUF model
def load_llm(model_path):
    return CTransformers(
        model=model_path,
        model_type="mistral",
        config={
            'max_new_tokens': 512,
            'temperature': 0.7,
            'context_length': 4096,
            'stream': False,
            'stop': ["</s>"]
        }
    )


# 🔸 Test CLI mode
if __name__ == "__main__":
    print("💬 Ready to query your documents.")
    query = input("Enter your question: ").strip()

    if not query:
        print("❌ No query entered. Exiting.")
        exit()

    # Paths
    base_dir = os.path.dirname(__file__)
    vector_path = os.path.abspath(os.path.join(base_dir, "..", "vectorstore"))
    model_path = r"C:\Users\Deepanshu\Desktop\RAG\AI_Resources\models\mistral-7b-instruct\mistral-7b-instruct-v0.1.Q4_K_M.gguf"

    # Load vectorstore and model
    db = load_vectorstore(vector_path)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = load_llm(model_path)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Run query
    result = qa_chain.invoke({"query": query})

    print("\n🧠 Answer:")
    print(result.get("result", "No answer found."))

    print("\n📄 Sources:")
    for i, doc in enumerate(result.get("source_documents", []), 1):
        page = doc.metadata.get("page_label", "unknown")
        content = doc.page_content.strip().replace("\n", " ")
        print(f"{i}. Page {page}: {content[:200]}...")
