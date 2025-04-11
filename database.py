from pymongo import MongoClient
import os

def get_db():
    client = MongoClient(os.getenv("MONGO_URI"))
    return client.rag_db.documents