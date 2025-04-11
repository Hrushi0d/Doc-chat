import datetime
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain.chains.retrieval_qa.base import RetrievalQA
from pydantic import BaseModel
import os
import uuid
from database import get_db
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from fastapi.responses import JSONResponse
from typing import Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Initialize services
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = OllamaLLM(model="deepseek-r1:8b", temperature=0.3)
db = get_db()

# File type handling
SUPPORTED_TYPES = {
    ".pdf": PyPDFLoader,
    ".txt": UnstructuredFileLoader,
    ".docx": UnstructuredFileLoader
}


class DocumentRequest(BaseModel):
    filename: str
    query: Optional[str] = None  # For query endpoint


class DocumentResponse(BaseModel):
    id: str
    filename: str
    processed: bool
    file_size: int
    created_at: datetime.datetime
    processed_at: Optional[datetime.datetime]
    chunk_count: Optional[int]


# --- Document Management Endpoints ---

@app.post("/documents", response_model=DocumentResponse)
async def add_document(file: UploadFile = File(...)):
    """Upload and process a document for RAG"""
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in SUPPORTED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {list(SUPPORTED_TYPES.keys())}"
        )

    os.makedirs("documents", exist_ok=True)
    file_uuid = uuid.uuid4()
    file_path = f"documents/{file_uuid}{file_ext}"
    doc_id = None

    try:
        # Save file in chunks
        with open(file_path, "wb") as f:
            while content := await file.read(1024 * 1024):  # 1MB chunks
                f.write(content)

        # Initial DB record
        doc_data = {
            "filename": file.filename,
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "file_type": file_ext[1:],
            "processed": False,
            "created_at": datetime.datetime.utcnow(),
            "processing_start": datetime.datetime.utcnow()
        }
        insert_result = db.insert_one(doc_data)
        doc_id = insert_result.inserted_id

        # Document processing
        try:
            loader = SUPPORTED_TYPES[file_ext](file_path)
            pages = loader.load_and_split()
        except Exception as e:
            logger.error(f"Document parsing failed: {e}", exc_info=True)
            raise HTTPException(422, "Document parsing failed")

        # Text splitting with metadata
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True
        )
        chunks = splitter.split_documents(pages)

        # Enhanced metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source_file": file.filename,
                "document_id": str(doc_id),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_type": file_ext[1:]
            })

        # Vector storage
        try:
            Pinecone.from_documents(
                documents=chunks,
                embedding=embeddings,
                index=pc.Index("rag-docs"),
                namespace=str(file_uuid),
                batch_size=50,
                show_progress=True
            )
        except Exception as e:
            logger.error(f"Pinecone error: {e}")
            raise HTTPException(503, "Vector storage service unavailable")

        # Update document status
        update_data = {
            "processed": True,
            "processed_at": datetime.datetime.utcnow(),
            "chunk_count": len(chunks),
            "processing_time": (
                    datetime.datetime.utcnow() - doc_data["processing_start"]
            ).total_seconds()
        }
        db.update_one({"_id": doc_id}, {"$set": update_data})

        return {
            "id": str(doc_id),
            "filename": file.filename,
            "processed": True,
            "file_size": doc_data["file_size"],
            "created_at": doc_data["created_at"],
            "processed_at": update_data["processed_at"],
            "chunk_count": update_data["chunk_count"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        # Cleanup
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        if doc_id:
            db.delete_one({"_id": doc_id})
        raise HTTPException(500, "Document processing failed")


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its vectors"""
    try:
        document = db.find_one({"_id": ObjectId(document_id)})
        if not document:
            raise HTTPException(404, "Document not found")

        # Delete from Pinecone
        index = pc.Index("rag-docs")
        index.delete(
            filter={
                "document_id": document_id,
                "source_file": document["filename"]
            }
        )

        # Delete local file
        if os.path.exists(document["file_path"]):
            os.remove(document["file_path"])

        # Delete from MongoDB
        db.delete_one({"_id": ObjectId(document_id)})

        return {"status": "success", "deleted_id": document_id}

    except Exception as e:
        logger.error(f"Deletion error: {e}")
        raise HTTPException(500, "Deletion failed")


@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents(limit: int = 100, skip: int = 0):
    """List all documents with pagination"""
    try:
        documents = []
        for doc in db.find().skip(skip).limit(limit):
            documents.append({
                "id": str(doc["_id"]),
                "filename": doc["filename"],
                "processed": doc.get("processed", False),
                "file_size": doc.get("file_size", 0),
                "created_at": doc.get("created_at"),
                "processed_at": doc.get("processed_at"),
                "chunk_count": doc.get("chunk_count")
            })
        return documents
    except Exception as e:
        logger.error(f"Listing error: {e}")
        raise HTTPException(500, "Failed to retrieve documents")


@app.post("/query")
async def query_documents(request: DocumentRequest):
    """Query a specific document"""
    try:
        document = db.find_one({"filename": request.filename})
        if not document:
            raise HTTPException(404, "Document not found")

        vectorstore = Pinecone(
            index=pc.Index("rag-docs"),
            embedding=embeddings,
            namespace=str(document["_id"])
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={
                    "k": 3,
                    "filter": {"document_id": str(document["_id"])}
                }
            )
        )
        result = qa_chain({"query": request.query})
        return {
            "response": result["result"],
            "source_document": request.filename,
            "document_id": str(document["_id"])
        }
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(500, "Query processing failed")