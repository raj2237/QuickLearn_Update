import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from redis.asyncio import Redis

load_dotenv()

class Config:
         SECRET_KEY = os.getenv('SECRET_KEY', 'quick')
         MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/quicklearnai')
         GROQ_API_KEY = os.getenv('GROQ_API_KEY')
         GENAI_API_KEY = os.getenv('GENAI_API_KEY')
         SERPER_API_KEY = os.getenv('SERPER_API_KEY')  # Remove hardcoded default
         HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
         PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
         PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
         SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
         YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
         # MongoDB setup
         mongo_client = AsyncIOMotorClient(MONGODB_URI)
         db = mongo_client['quicklearnai']
         topics_collection = db['statistics']
         
         # Redis setup
         async def get_redis_client():
             return await Redis.from_url('redis://localhost:6379/0', decode_responses=True)
         
         # Folder paths
         UPLOAD_FOLDER = 'Uploads'
         OUTPUT_FOLDER = 'generated_papers'