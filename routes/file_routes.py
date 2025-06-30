from flask import Blueprint, request, jsonify, send_file
from utils.pdf_utils import extract_text_from_pdf, extract_text_from_pptx
from utils.tts_utils import tts_manager, clean_response
from config import Config
import os
import aiofiles
import google.generativeai as genai
import logging
from pinecone import Pinecone, ServerlessSpec

file_bp = Blueprint('file', __name__)

# Initialize Pinecone
pinecone_client = Pinecone(api_key=Config.PINECONE_API_KEY)
INDEX_NAME = "quicklearn"

# Initialize Gemini
genai.configure(api_key=Config.GENAI_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def initialize_pinecone_index():
    try:
        # Check if index exists, create if it doesn't
        if INDEX_NAME not in pinecone_client.list_indexes().names():
            pinecone_client.create_index(
                name=INDEX_NAME,
                dimension=1024,  # Gemini text-embedding-004 dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=Config.PINECONE_ENVIRONMENT)
            )
        return pinecone_client.Index(INDEX_NAME)
    except Exception as e:
        logger.error(f"Error initializing Pinecone index: {str(e)}")
        raise

@file_bp.route("/upload", methods=["POST"])
async def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    file_ext = os.path.splitext(file.filename)[-1].lower()
    file_path = os.path.join(Config.UPLOAD_FOLDER, file.filename)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(file.read())
    
    try:
        if file_ext == ".pdf":
            content = await extract_text_from_pdf(file_path)
        elif file_ext == ".pptx":
            content = await extract_text_from_pptx(file_path)
        else:
            return jsonify({"error": "Unsupported file format. Only PDF and PPTX are allowed."}), 400
        
        # Generate embedding using Gemini
        embedding_response = await genai.embed_content_async(
            model="models/text-embedding-004",
            content=content,
            task_type="retrieval_document"
        )
        embedding = embedding_response.get("embedding", [])
        if not embedding:
            return jsonify({"error": "Failed to generate embedding"}), 500
        
        # Store in Pinecone
        index = await initialize_pinecone_index()
        index.upsert(vectors=[(file.filename, embedding, {"text": content})])
        
        return jsonify({"message": "File uploaded and processed successfully."}), 200
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({"error": str(e)}), 500

@file_bp.route("/test-audio", methods=["GET"])
async def test_audio():
    try:
        test_text = "This is a test of the text to speech system"
        logger.info("Testing text-to-speech with test message")
        
        speech_thread = tts_manager.start_speaking(test_text)
        
        return jsonify({
            "message": "Audio test initiated",
            "test_text": test_text,
            "status": "Speech initiated"
        })
    except Exception as e:
        logger.error(f"Audio test failed: {str(e)}")
        return jsonify({
            "error": "Audio test failed",
            "details": str(e)
        }), 500

@file_bp.route("/query", methods=["POST"])
async def query_file():
    try:
        data = request.get_json()
        query = data.get("query", "")
        
        logger.info(f"Received query: {query}")
        
        # Generate query embedding using Gemini
        query_embedding_response = await genai.embed_content_async(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = query_embedding_response.get("embedding", [])
        if not query_embedding:
            return jsonify({"error": "Failed to generate query embedding"}), 500
        
        # Query Pinecone
        index = await initialize_pinecone_index()
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        retrieved_texts = "\n".join([match["metadata"]["text"] for match in results["matches"]])
        
        prompt = f"""
        Based on the following context, please provide a clear and concise answer to the question.
        If the answer cannot be found in the context, please say so.
        
        Context: {retrieved_texts}
        
        Question: {query}
        """
        
        response = await genai.GenerativeModel("gemini-1.5-flash").generate_content_async(prompt)
        cleaned_response = clean_response(response.text)
        
        speech_thread = tts_manager.start_speaking(cleaned_response)
        
        return jsonify({
            "answer": cleaned_response,
            "voice_enabled": True,
            "status": "Speech initiated"
        })
        
    except Exception as e:
        error_message = f"Error processing query: {str(e)}"
        logger.error(error_message)
        return jsonify({
            "error": error_message,
            "answer": "I apologize, but I encountered an error while processing your query. Please try again.",
            "voice_enabled": False
        }), 500