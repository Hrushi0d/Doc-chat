import datetime
import os
import uuid
from typing import Optional, List

import torch
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM as Ollama
from pinecone import Pinecone
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

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


def process_document(file_path: str, file_ext: str) -> dict:
    """
    Function to process a document, split it into chunks, store embeddings in Pinecone,
    and save metadata in MongoDB with detailed error handling.
    """
    doc_id = None
    try:
        # Create a unique ID for the document
        file_uuid = str(uuid.uuid4())

        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        # Load the document based on its type
        try:
            loader = SUPPORTED_TYPES[file_ext](file_path)
            pages = loader.load()
        except ValueError as e:
            raise ValueError(f"Failed to load document {file_path}: {e}")
        except Exception as e:
            raise Exception(f"Error loading file {file_path}: {str(e)}")

        # Split the document into smaller chunks
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                add_start_index=True
            )
            chunks = splitter.split_documents(pages)
        except Exception as e:
            raise Exception(f"Error splitting document into chunks: {str(e)}")

        # Add chunks to Pinecone vectorstore
        try:
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "source_file": os.path.basename(file_path),
                    "document_id": file_uuid,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_type": file_ext[1:]
                })

            vectorstore.add_documents(documents=chunks, namespace="ollamarag1")
        except Exception as e:
            raise Exception(f"Pinecone error: {str(e)}")

        # Store metadata in MongoDB
        try:
            doc_data = {
                "filename": os.path.basename(file_path),
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "file_type": file_ext[1:],
                "processed": True,
                "created_at": datetime.datetime.utcnow(),
                "processed_at": datetime.datetime.utcnow(),
                "chunk_count": len(chunks),
            }
            insert_result = db.insert_one(doc_data)
            doc_id = str(insert_result.inserted_id)
        except PyMongoError as e:
            raise Exception(f"MongoDB error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error inserting metadata into MongoDB: {str(e)}")

        return {
            "id": doc_id,
            "filename": os.path.basename(file_path),
            "processed": True,
            "file_size": doc_data["file_size"],
            "created_at": doc_data["created_at"],
            "processed_at": doc_data["processed_at"],
            "chunk_count": doc_data["chunk_count"]
        }

    except Exception as e:
        logger.error(f"Document processing failed: {e}", exc_info=True)

        # Clean up: remove file and delete doc from MongoDB if any
        if os.path.exists(file_path):
            os.remove(file_path)
        if doc_id:
            try:
                db.delete_one({"_id": ObjectId(doc_id)})
            except Exception as delete_error:
                logger.error(f"Error deleting document from MongoDB: {delete_error}")

        raise Exception(f"Document processing failed: {str(e)}")


# Testing function (use a local file for testing)
if __name__ == "__main__":
    local_file_path = "2021.nlp4convai-1.14.pdf"  # Provide the path to your local file
    file_ext = os.path.splitext(local_file_path)[1].lower()

    if file_ext not in SUPPORTED_TYPES:
        logger.error(f"Unsupported file type. Supported types are: {list(SUPPORTED_TYPES.keys())}")
    else:
        try:
            document_info = process_document(local_file_path, file_ext)
            logger.info(f"Document processed successfully: {document_info}")
        except Exception as e:
            logger.error(f"Error processing document: {e}")
