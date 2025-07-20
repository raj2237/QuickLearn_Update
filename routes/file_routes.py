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
from nomic import embed
import numpy as np

file_bp = Blueprint('file', __name__)

# Initialize Pinecone
pinecone_client = Pinecone(api_key=Config.PINECONE_API_KEY)
INDEX_NAME = "quicklearn"

# Initialize Gemini (only for text generation, not embeddings)
genai.configure(api_key=Config.GENAI_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def async_route(f):
    """Decorator to handle async routes in Flask"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            # Create a new event loop if none exists or it's closed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        except Exception as e:
            logger.error(f"Error in async route: {str(e)}")
            raise
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
    """Initialize Pinecone index with dimension for Nomic embeddings."""
    try:
        if INDEX_NAME not in pinecone_client.list_indexes().names():
            pinecone_client.create_index(
                name=INDEX_NAME,
                dimension=768,  # Nomic embed-text-v1.5 uses 768 dimensions
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=Config.PINECONE_ENVIRONMENT)
            )
        return pinecone_client.Index(INDEX_NAME)
    except Exception as e:
        logger.error(f"Error initializing Pinecone index: {str(e)}")
        raise

def generate_nomic_embeddings_sync(texts, task_type="search_document"):
    """Generate embeddings using Nomic AI - synchronous version."""
    try:
        logger.info(f"Generating embeddings for {len(texts)} texts using Nomic AI")
        start_time = time.time()
        
        # Use Nomic's embed.text function
        response = embed.text(
            texts=texts,
            model='nomic-embed-text-v1.5',
            task_type=task_type,
            dimensionality=768  # Explicitly set dimensionality
        )
        
        embeddings = response['embeddings']
        logger.info(f"Nomic embedding generation took {time.time() - start_time:.2f} seconds")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating Nomic embeddings: {str(e)}")
        return [None] * len(texts)

async def batch_embed_chunks(chunks, batch_size=32):
    """Generate embeddings for chunks in batches using Nomic AI."""
    start_time = time.time()
    embeddings = []
    
    # Filter out empty or too short chunks
    valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 10]
    
    for i in range(0, len(valid_chunks), batch_size):
        batch = valid_chunks[i:i + batch_size]
        try:
            # Use thread pool executor to run the sync function
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(generate_nomic_embeddings_sync, batch, "search_document")
                batch_embeddings = future.result()
            embeddings.extend(batch_embeddings)
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {str(e)}")
            embeddings.extend([None] * len(batch))
            
        # Add small delay to avoid rate limiting
        await asyncio.sleep(0.1)
    
    logger.info(f"Total embedding generation took {time.time() - start_time:.2f} seconds")
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
        
        # Generate embeddings using Nomic AI
        index = await initialize_pinecone_index()
        embeddings = await batch_embed_chunks(chunks)
        
        # Prepare and upsert vectors
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding is not None and len(chunk.strip()) >= 50:
                # Convert embedding to list if it's a numpy array
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                vectors_to_upsert.append({
                    "id": f"{file.filename}_{i}",
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "filename": file.filename,
                        "chunk_index": i,
                        "full_text": content
                    }
                })
        
        # Upsert vectors in batches
        for i in range(0, len(vectors_to_upsert), 100):
            batch = vectors_to_upsert[i:i + 100]
            index.upsert(vectors=batch)
        
        logger.info(f"Upload took {time.time() - start_time:.2f} seconds")
        return jsonify({
            "message": "File uploaded and processed successfully with Nomic AI embeddings.",
            "chunks_created": len(vectors_to_upsert)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

@file_bp.route("/query", methods=["POST"])
@async_route
async def query_file():
    """Query uploaded files and generate response with TTS using Nomic embeddings."""
    start_time = time.time()
    try:
        data = request.get_json()
        query = data.get("query", "")
        if not query:
            return jsonify({"error": "No query provided"}), 400

        logger.info(f"Received query: {query}")
        
        # Generate query embedding using Nomic AI
        import concurrent.futures
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(generate_nomic_embeddings_sync, [query], "search_query")
                query_embeddings = future.result()
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return jsonify({"error": "Failed to generate query embedding"}), 500
        
        if not query_embeddings or query_embeddings[0] is None:
            return jsonify({"error": "Failed to generate query embedding"}), 500
        
        query_embedding = query_embeddings[0]
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        index = await initialize_pinecone_index()
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        
        if not results["matches"]:
            return jsonify({
                "answer": "No relevant content found in the uploaded documents.",
                "voice_enabled": False,
                "context": []
            }), 404

        # Get the best match
        best_match = results["matches"][0]
        chunk_text = best_match["metadata"]["text"]
        filename = best_match["metadata"].get("filename", "Unknown source")
        full_text = best_match["metadata"].get("full_text", "")
        
        # Get additional context from multiple matches
        context_chunks = [match["metadata"]["text"] for match in results["matches"][:3]]
        combined_context = "\n\n".join(context_chunks)
        
        context_lines = get_chunk_context(full_text, chunk_text, context_lines=2)
        if len(context_lines) > 4:
            context_lines = context_lines[:4]
        
        prompt = f"""
        Answer the question based on the context provided. If the answer is not in the context, say so clearly.
        Be concise and accurate in your response.
        
        Context: {combined_context}
        
        Question: {query}
        
        Answer:
        """
        
        response = await genai.GenerativeModel("gemini-2.5-flash").generate_content_async(prompt)
        cleaned_response = clean_response(response.text)
        
        logger.info("Generating speech for response...")
        tts_thread = tts_manager.start_speaking(cleaned_response)
        tts_thread.join(timeout=0.1)  # Brief wait to ensure thread starts
        voice_enabled = tts_thread.is_alive()  # Check if TTS thread is running
        
        logger.info(f"Query took {time.time() - start_time:.2f} seconds")
        return jsonify({
            "answer": cleaned_response,
            "voice_enabled": voice_enabled,
            "context": {
                "source": filename,
                "relevant_lines": context_lines,
                "relevance_score": best_match["score"],
                "matches_found": len(results["matches"])
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

@file_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "embedding_model": "nomic-embed-text-v1.5",
        "index_name": INDEX_NAME
    }), 200