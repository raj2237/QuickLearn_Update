import os
from dotenv import load_dotenv
from pymongo import MongoClient
import redis

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'quick')
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/quicklearnai')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GENAI_API_KEY = os.getenv('GENAI_API_KEY')
    SERPER_API_KEY = os.getenv('SERPER_API_KEY', '85a684d9cfcddab4886460954ef36f054053529b')
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
    
    # MongoDB setup
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client['quicklearnai']
    topics_collection = db['statistics']
    
    # Redis setup
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
    
    # Folder paths
    UPLOAD_FOLDER = 'Uploads'
    OUTPUT_FOLDER = 'generated_papers'