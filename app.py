import datetime
import logging
import os
import uuid
from typing import Optional, List

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM as Ollama
from pinecone import Pinecone
from pydantic import BaseModel
from bson import ObjectId
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Initialize CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')

# Initialize Pinecone
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
llm = Ollama(model="deepseek-r1:1.5b", temperature=0.3)


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


# Request/Response models
class DocumentRequest(BaseModel):
    filename: str
    query: Optional[str] = None


class DocumentResponse(BaseModel):
    id: str
    filename: str
    processed: bool
    file_size: int
    created_at: datetime.datetime
    processed_at: Optional[datetime.datetime] = None
    chunk_count: Optional[int] = None


class QueryRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None
    k: Optional[int] = 10


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


@app.post("/documents", response_model=DocumentResponse)
async def add_document(file: UploadFile = File(...)):
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in SUPPORTED_TYPES:
        raise HTTPException(400, f"Unsupported file type. Supported: {list(SUPPORTED_TYPES.keys())}")

    os.makedirs("documents", exist_ok=True)
    file_uuid = str(uuid.uuid4())
    file_path = f"documents/{file_uuid}{file_ext}"
    doc_id = None

    try:
        with open(file_path, "wb") as f:
            while content := await file.read(1024 * 1024):
                f.write(content)

        doc_data = {
            "filename": file.filename,
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "file_type": file_ext[1:],
            "processed": False,
            "created_at": datetime.datetime.utcnow(),
            "processing_start": datetime.datetime.utcnow()
        }
        insert_result = db.documents.insert_one(doc_data)
        doc_id = str(insert_result.inserted_id)

        loader = SUPPORTED_TYPES[file_ext](file_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True
        )
        chunks = splitter.split_documents(pages)

        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source_file": file.filename,
                "document_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_type": file_ext[1:]
            })

        vectorstore.add_documents(documents=chunks, namespace=file_uuid)

        update_data = {
            "processed": True,
            "processed_at": datetime.datetime.utcnow(),
            "chunk_count": len(chunks),
            "processing_time": (
                    datetime.datetime.utcnow() - doc_data["processing_start"]
            ).total_seconds()
        }
        db.documents.update_one({"_id": ObjectId(doc_id)}, {"$set": update_data})

        return DocumentResponse(
            id=doc_id,
            filename=file.filename,
            processed=True,
            file_size=doc_data["file_size"],
            created_at=doc_data["created_at"],
            processed_at=update_data["processed_at"],
            chunk_count=update_data["chunk_count"]
        )

    except Exception as e:
        logger.error(f"Document processing failed: {e}", exc_info=True)
        if os.path.exists(file_path):
            os.remove(file_path)
        if doc_id:
            db.documents.delete_one({"_id": ObjectId(doc_id)})
        raise HTTPException(500, f"Document processing failed: {str(e)}")


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    try:
        try:
            obj_id = ObjectId(document_id)
        except:
            raise HTTPException(400, "Invalid document ID format")

        document = db.documents.find_one({"_id": obj_id})
        if not document:
            raise HTTPException(404, "Document not found")

        pc.Index("rag-docs").delete(
            namespace=os.path.splitext(os.path.basename(document["file_path"]))[0],
            filter={"document_id": document_id}
        )

        if os.path.exists(document["file_path"]):
            os.remove(document["file_path"])

        db.documents.delete_one({"_id": obj_id})

        return {"status": "success", "deleted_id": document_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deletion failed: {e}", exc_info=True)
        raise HTTPException(500, f"Deletion failed: {str(e)}")


@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents(limit: int = 100, skip: int = 0):
    try:
        documents = []
        for doc in db.documents.find().skip(skip).limit(limit):
            documents.append(DocumentResponse(
                id=str(doc["_id"]),
                filename=doc["filename"],
                processed=doc.get("processed", False),
                file_size=doc.get("file_size", 0),
                created_at=doc.get("created_at"),
                processed_at=doc.get("processed_at"),
                chunk_count=doc.get("chunk_count")
            ))
        return documents
    except Exception as e:
        logger.error(f"Listing failed: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to retrieve documents: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        filter_dict = None
        if request.document_ids:
            try:
                [ObjectId(doc_id) for doc_id in request.document_ids]
                filter_dict = {"document_id": {"$in": request.document_ids}}
            except:
                raise HTTPException(400, "Invalid document ID format")

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": request.k or 5,
                "filter": filter_dict
            }
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        result = qa({"query": request.question})

        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "document_id": doc.metadata.get("document_id"),
                "source_file": doc.metadata.get("source_file"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "page_content": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
            })

        return QueryResponse(
            answer=result.get("result", "No answer found"),
            sources=sources
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question answering failed: {e}", exc_info=True)
        raise HTTPException(500, f"Question answering failed: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
