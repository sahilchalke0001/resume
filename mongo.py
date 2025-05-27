import os
from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()

mongo_uri = os.getenv("MONGO_URI")

# Connect to MongoDB Atlas
client = MongoClient(mongo_uri)
db = client["db"]
collection = db["knowledge_chunks"]

with open("knowledge_base/1.txt", "r", encoding="utf-8") as file:
    full_content = file.read()
    collection.insert_one({"content": full_content})
