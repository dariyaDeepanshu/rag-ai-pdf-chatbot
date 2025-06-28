# # === src/main.py ===
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from src.query import load_vectorstore, load_llm
# from langchain.chains import RetrievalQA
# import os

# app = FastAPI(title="RAG AI Assistant")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# class QueryRequest(BaseModel):
#     query: str


# VECTOR_PATH = os.path.join("vectorstore")
# MODEL_PATH = r"C:\\Users\\Deepanshu\\Desktop\\RAG\\AI_Resources\\models\\mistral-7b-instruct\\mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# db = load_vectorstore(VECTOR_PATH)
# retriever = db.as_retriever(search_kwargs={"k": 3})
# llm = load_llm(MODEL_PATH)

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     return_source_documents=True
# )


# @app.post("/query")
# async def query(req: QueryRequest):
#     try:
#         result = qa_chain.invoke({"query": req.query})
#         answer_text = result.get("result", "").strip()
#         sources = []
#         for doc in result.get("source_documents", []):
#             page = doc.metadata.get("page_label", "unknown")
#             content = doc.page_content.strip().replace("\n", " ")
#             sources.append(f"Page {page}: {content[:300]}...")
#         return {"answer": answer_text, "sources": sources}
#     except Exception as e:
#         return {"answer": f"Error: {str(e)}", "sources": []}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.query import load_vectorstore, load_llm
from langchain.chains import RetrievalQA
import os

# 🔧 Setup FastAPI app
app = FastAPI(title="RAG AI Assistant")

# 🔓 Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📬 Request schema


class QueryRequest(BaseModel):
    query: str


# 📂 Paths
VECTOR_PATH = os.path.join("vectorstore")
MODEL_PATH = r"C:\Users\Deepanshu\Desktop\RAG\AI_Resources\models\mistral-7b-instruct\mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# 🧠 Load vectorstore and model
print("📦 Loading vectorstore and model...")
db = load_vectorstore(VECTOR_PATH)
retriever = db.as_retriever(search_kwargs={"k": 5})
llm = load_llm(MODEL_PATH)

# 🔗 Build RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 🚀 Query endpoint


@app.post("/query")
async def query(req: QueryRequest):
    try:
        result = qa_chain.invoke({"query": req.query})

        # ✅ FIXED: Access text inside the dict
        answer_text = result.get("result", "").strip()

        # 📄 Format source chunks
        sources = []
        for doc in result.get("source_documents", []):
            page = doc.metadata.get("page_label", "unknown")
            content = doc.page_content.strip().replace("\n", " ")
            sources.append(f"Page {page}: {content[:300]}...")

        return {
            "answer": answer_text,
            "sources": sources
        }

    except Exception as e:
        print(f"❌ Query error: {e}")
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }
