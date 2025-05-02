import logging
import os
import time
from typing import Optional, List

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM as Ollama
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pydantic import BaseModel
from pymongo import MongoClient
from starlette.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')

# Initialize Pinecone Vector Database
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-docs")

# HuggingFace embeddings with GPU support
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
    encode_kwargs={"device": device, "batch_size": 32}
)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# Ollama model initialization
llms = {
    'deepseek-r1': Ollama(model="deepseek-r1:1.5b", temperature=0.3),
    'deepseek-coder': Ollama(model="deepseek-coder", temperature=0.3)
}

# MongoDB connection
def get_db():
    client = MongoClient(os.getenv("MONGO_URI"))
    return client.rag_db.documents


db = get_db()

# Supported file types
SUPPORTED_TYPES = {
    ".pdf": PyPDFLoader,
    ".txt": UnstructuredFileLoader,
    ".docx": UnstructuredFileLoader
}


class QueryRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None
    k: Optional[int] = 10
    model: Optional[str] = "deepseek-r1"


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    model: str
    # time: str


@app.post("/query", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    # start_time = time.time()
    try:
        filter_dict = None
        if request.document_ids:
            try:
                filter_dict = {"document_id": {"$in": request.document_ids}}
            except Exception as e:
                raise HTTPException(400, "Invalid document ID format")

        search_kwargs = {
            "k": request.k or 5,
            "namespace": "ollamarag1"
        }
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

        qa = RetrievalQA.from_chain_type(
            llm=llms[request.model],
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        result = qa.invoke({"query": request.question})

        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "document_id": doc.metadata.get("document_id"),
                "source_file": doc.metadata.get("source_file"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "page_content": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
            })
        # end_time = time.time()
        # time_elapsed = f"Execution completed in {end_time - start_time:.2f} seconds"
        return QueryResponse(
            answer=result.get("result", "No answer found"),
            sources=sources,
            model=request.model
            # ,time=time_elapsed
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question answering failed: {e}", exc_info=True)
        raise HTTPException(500, f"Question answering failed: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
