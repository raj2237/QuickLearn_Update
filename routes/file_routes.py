from flask import Blueprint, request, jsonify
from utils.pdf_utils import extract_text_from_pdf, extract_text_from_pptx
from utils.tts_utils import tts_manager, clean_response
from config import Config
import os
import aiofiles
import google.generativeai as genai
import logging
from pinecone import Pinecone, ServerlessSpec
import re
import time
import asyncio
from functools import wraps

file_bp = Blueprint('file', __name__)

# Initialize Pinecone
pinecone_client = Pinecone(api_key=Config.PINECONE_API_KEY)
INDEX_NAME = "quicklearn"

# Initialize Gemini
genai.configure(api_key=Config.GENAI_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def async_route(f):
    """Decorator to handle async routes in Flask"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

def chunk_text(text, chunk_size=500, overlap=50, max_chunks=50):
    """Split text into overlapping chunks."""
    start_time = time.time()
    words = text.strip().split()
    chunks = []
    current_chunk = []
    chunk_count = 0

    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) > chunk_size:
            chunks.append(' '.join(current_chunk))
            chunk_count += 1
            if chunk_count >= max_chunks:
                break
            current_chunk = current_chunk[-overlap:]  # Overlap for continuity
        if len(chunks) == max_chunks:
            break

    if current_chunk and chunk_count < max_chunks:
        chunks.append(' '.join(current_chunk))

    logger.info(f"Chunking took {time.time() - start_time:.2f} seconds")
    return chunks

def get_chunk_context(text, chunk_text, context_lines=2):
    """Get surrounding context lines for a specific chunk."""
    lines = text.split('\n')
    chunk_lines = chunk_text.split('\n')
    
    start_line = -1
    for i, line in enumerate(lines):
        if chunk_lines[0].strip() in line.strip():
            start_line = i
            break
    
    if start_line == -1:
        return chunk_lines[:context_lines * 2]
    
    context_start = max(0, start_line - context_lines)
    context_end = min(len(lines), start_line + len(chunk_lines) + context_lines)
    
    return lines[context_start:context_end]

async def initialize_pinecone_index():
    """Initialize Pinecone index."""
    try:
        if INDEX_NAME not in pinecone_client.list_indexes().names():
            pinecone_client.create_index(
                name=INDEX_NAME,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=Config.PINECONE_ENVIRONMENT)
            )
        return pinecone_client.Index(INDEX_NAME)
    except Exception as e:
        logger.error(f"Error initializing Pinecone index: {str(e)}")
        raise

async def batch_embed_chunks(chunks, batch_size=5):
    """Generate embeddings for chunks in batches asynchronously."""
    start_time = time.time()
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            tasks = [
                genai.embed_content_async(
                    model="models/text-embedding-004",
                    content=chunk,
                    task_type="retrieval_document"
                ) for chunk in batch
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for response in responses:
                if isinstance(response, Exception):
                    logger.error(f"Embedding error: {str(response)}")
                    embeddings.append(None)
                else:
                    embeddings.append(response.get("embedding", None))
        except Exception as e:
            logger.warning(f"Embedding failed: {str(e)}")
            embeddings.extend([None] * len(batch))
    logger.info(f"Embedding generation took {time.time() - start_time:.2f} seconds")
    return embeddings

@file_bp.route("/upload", methods=["POST"])
@async_route
async def upload_file():
    """Upload and process PDF or PPTX files."""
    start_time = time.time()
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No valid file provided"}), 400

    file = request.files["file"]
    file_ext = os.path.splitext(file.filename)[-1].lower()
    if file_ext not in (".pdf", ".pptx"):
        return jsonify({"error": "Unsupported file format. Only PDF and PPTX are allowed."}), 400

    file_path = os.path.join(Config.UPLOAD_FOLDER, file.filename)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(file.read())

    try:
        # Extract text
        content = await (extract_text_from_pdf(file_path) if file_ext == ".pdf" else extract_text_from_pptx(file_path))
        
        # Split content into chunks
        chunks = chunk_text(content)
        
        # Generate embeddings
        index = await initialize_pinecone_index()
        embeddings = await batch_embed_chunks(chunks)
        
        # Prepare and upsert vectors
        vectors_to_upsert = [
            (f"{file.filename}_{i}", embedding, {
                "text": chunk,
                "filename": file.filename,
                "chunk_index": i,
                "full_text": content
            })
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            if embedding and len(chunk.strip()) >= 50
        ]
        
        for i in range(0, len(vectors_to_upsert), 100):
            index.upsert(vectors=vectors_to_upsert[i:i + 100])
        
        logger.info(f"Upload took {time.time() - start_time:.2f} seconds")
        return jsonify({
            "message": "File uploaded and processed successfully.",
            "chunks_created": len(vectors_to_upsert)
        }), 200
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({"error": str(e)}), 500

@file_bp.route("/query", methods=["POST"])
@async_route
async def query_file():
    """Query uploaded files and generate response with TTS."""
    start_time = time.time()
    try:
        data = request.get_json()
        query = data.get("query", "")
        if not query:
            return jsonify({"error": "No query provided"}), 400

        logger.info(f"Received query: {query}")
        
        query_embedding_response = await genai.embed_content_async(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = query_embedding_response.get("embedding", [])
        if not query_embedding:
            return jsonify({"error": "Failed to generate query embedding"}), 500

        index = await initialize_pinecone_index()
        results = index.query(vector=query_embedding, top_k=1, include_metadata=True)
        
        if not results["matches"]:
            return jsonify({
                "answer": "No relevant content found in the uploaded documents.",
                "voice_enabled": False,
                "context": []
            }), 404

        best_match = results["matches"][0]
        chunk_text = best_match["metadata"]["text"]
        filename = best_match["metadata"].get("filename", "Unknown source")
        full_text = best_match["metadata"].get("full_text", "")
        
        context_lines = get_chunk_context(full_text, chunk_text, context_lines=2)
        if len(context_lines) > 4:
            context_lines = context_lines[:4]
        
        prompt = f"""
        Answer the question based on the context. If the answer is not in the context, say so.
        Context: {chunk_text}
        Question: {query}
        """
        
        response = await genai.GenerativeModel("gemini-1.5-flash").generate_content_async(prompt)
        cleaned_response = clean_response(response.text)
        
        logger.info("Generating speech for response...")
        # Start TTS in a thread without awaiting
        tts_thread = tts_manager.start_speaking(cleaned_response)
        # Optionally wait briefly to ensure thread starts
        tts_thread.join(timeout=0.1)  # Non-blocking, allows thread to start
        
        logger.info(f"Query took {time.time() - start_time:.2f} seconds")
        return jsonify({
            "answer": cleaned_response,
            "voice_enabled": True,
            "context": {
                "source": filename,
                "relevant_lines": context_lines,
                "relevance_score": best_match["score"]
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({
            "error": str(e),
            "answer": "An error occurred while processing your query. Please try again.",
            "voice_enabled": False,
            "context": []
        }), 500

@file_bp.route("/clear_embeddings", methods=["POST"])
@async_route
async def clear_embeddings():
    """Clear all embeddings from the Pinecone index."""
    try:
        index = await initialize_pinecone_index()
        index.delete(delete_all=True)
        logger.info("All embeddings cleared from Pinecone index")
        return jsonify({"message": "All embeddings cleared from Pinecone."}), 200
    except Exception as e:
        logger.error(f"Error clearing embeddings: {str(e)}")
        return jsonify({"error": str(e)}), 500