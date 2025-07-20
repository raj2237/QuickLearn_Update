from flask import Blueprint, request, jsonify
from utils.auth_utils import validate_token_middleware
from config import Config
from langchain_groq import ChatGroq
import json
import re
from duckduckgo_search import DDGS
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp
recommendation_bp = Blueprint('recommendation', __name__)

@recommendation_bp.route('/getonly', methods=['GET'])
@validate_token_middleware()
async def get_recommendations():
    user_id = request.user_id
    
    try:
        redis_client = await Config.get_redis_client()
        statistics = await redis_client.hget(f"student:{user_id}", "statistics")
        
        if not statistics:
            return jsonify({"message": "No statistics found for the provided user."}), 404
        
        topics_data = json.loads(statistics)
        if not topics_data:
            return jsonify({"message": "No topics found for the provided user."}), 404

        topics_list = list(topics_data.keys())

        prompt = f"""
        Act as an intelligent recommendation generator. Based on the topics provided, generate a structured JSON response 
        with an overview, recommendations, and five YouTube video URLs for each topic. Ensure the output is in strict JSON 
        format without markdown or extra formatting. Use the following JSON structure:
        {{
            "topics": {{
                "<topic_name>": {{
                    "overview": "<brief overview>",
                    "recommendations": "<recommended steps to learn>",
                    "youtube_links": [
                        "<video_link_1>",
                        "<video_link_2>",
                        "<video_link_3>",
                        "<video_link_4>",
                        "<video_link_5>"
                    ]
                }}
            }}
        }}

        The topics are: {', '.join(topics_list)}
        """

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=Config.GROQ_API_KEY
        )
        
        response = await llm.ainvoke(prompt)
        recommendations_raw = response.content if hasattr(response, 'content') else str(response)

        try:
            recommendations = json.loads(recommendations_raw)
        except json.JSONDecodeError:
            return jsonify({"message": "Failed to parse AI response as JSON", "raw_response": recommendations_raw}), 500

        return jsonify({
            "message": "Recommendations generated successfully",
            "recommendations": recommendations["topics"]
        }), 200

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

@recommendation_bp.route('/youtube_videos', methods=['POST', 'OPTIONS'])
async def youtube_videos():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.json
        if not data or 'topic' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'topic' in JSON body"
            }), 400
        
        topics = data['topic']
        if isinstance(topics, str):
            topics = [topics]
        
        if not isinstance(topics, list) or not topics:
            return jsonify({
                "success": False,
                "error": "'topic' must be a non-empty list"
            }), 400
        
        result = {}
        for topic in topics:
            video_urls = await search_youtube_videos(topic, max_results=3)
            valid_urls = [url for url in video_urls if is_valid_youtube_url(url)]
            result[topic] = valid_urls[:3]
        
        return jsonify({
            "success": True,
            "data": result
        })
    
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": "Invalid JSON format"
        }), 400
    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

async def search_youtube_videos(topic, max_results=3):
    """
    Search for YouTube videos using YouTube Data API v3
    """
    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            'part': 'snippet',
            'q': f"{topic} tutorial",
            'type': 'video',
            'maxResults': max_results,
            'key': Config.YOUTUBE_API_KEY,
            'order': 'relevance',
            'videoDefinition': 'any',
            'videoDuration': 'any'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                video_urls = []
                if 'items' in data:
                    for item in data['items']:
                        video_id = item['id']['videoId']
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        video_urls.append(video_url)
                
                return video_urls
                
    except Exception as e:
        print(f"YouTube API error for {topic}: {e}")
        return []

def is_valid_youtube_url(url):
    """
    Validate if the URL is a valid YouTube video URL
    """
    patterns = [
        r'^https?://(?:www\.)?youtube\.com/watch\?v=[\w-]{11}',
        r'^https?://youtu\.be/[\w-]{11}',
        r'^https?://(?:www\.)?youtube\.com/embed/[\w-]{11}',
        r'^https?://(?:m\.)?youtube\.com/watch\?v=[\w-]{11}'
    ]
    
    return any(re.match(pattern, url) for pattern in patterns)