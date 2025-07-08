# backend/db/mongo_client.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "medassist_db")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]  # ✅ This is the only thing you should import from here
