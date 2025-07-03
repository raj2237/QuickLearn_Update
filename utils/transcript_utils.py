import aiohttp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from langchain_groq import ChatGroq
import google.generativeai as genai
import re
import json
from config import Config
import asyncio

formatter = TextFormatter()

async def get_and_enhance_transcript(youtube_url, model_type='gemini'):
    try:
        # Extract video ID with proper handling of query parameters
        video_id = re.search(r'(?:v=|youtu\.be/)([^&\n?#]+)', youtube_url)
        if not video_id:
            print(f"Invalid YouTube URL: {youtube_url}")
            return None, None
        video_id = video_id.group(1)

        # Fetch transcript using YouTubeTranscriptApi
        transcript = None
        language = None
        for lang in ['en', 'hi']:
            try:
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                transcript = formatter.format_transcript(transcript_data)
                language = lang
                break
            except Exception as e:
                print(f"Failed to fetch transcript for language {lang}: {str(e)}")
                continue

        if not transcript:
            print(f"No transcript available for video ID: {video_id}")
            return None, None

        # Enhance transcript
        prompt = f"""
        Act as a transcript cleaner. Generate a new transcript with the same context and content as the given transcript.
        If there's a revision portion, differentiate it from the actual transcript.
        Output in sentences line by line. If the transcript lacks educational content, return 'Fake transcript'.
        Transcript: {transcript}
        """

        if model_type.lower() == 'chatgroq':
            groq_model = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                groq_api_key=Config.GROQ_API_KEY
            )
            response = await groq_model.ainvoke(prompt)
            enhanced_transcript = response.content if hasattr(response, 'content') else str(response)
        else:
            genai.configure(api_key=Config.GENAI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            response = await gemini_model.generate_content_async(prompt)
            enhanced_transcript = response.text if hasattr(response, 'text') else str(response)

        return enhanced_transcript, language
    except Exception as e:
        print(f"Error in get_and_enhance_transcript: {str(e)}")
        return None, None

async def generate_summary_and_quiz(transcript, num_questions, language, difficulty, model_type='gemini'):
    try:
        if 'Fake transcript' in transcript:
            return {"summary": {}, "questions": {difficulty: []}}

        prompt = f"""
        Summarize the following transcript by identifying the key topics covered, and provide a detailed summary of each topic in 6-7 sentences.
        Each topic should be labeled clearly as "Topic X", where X is the topic name. Provide the full summary for each topic in English, even if the transcript is in a different language.
        Strictly ensure that possessives (e.g., John's book) and contractions (e.g., don't) use apostrophes (') instead of quotation marks (" or "  ").

        If the transcript contains 'Fake Transcript', do not generate any quiz or summary.

        After the summary, give the name of the topic on which the transcript was all about in a maximum of 2 to 3 words.
        After summarizing, create a quiz with {num_questions} multiple-choice questions in English, based on the transcript content.
        Only generate {difficulty} difficulty questions. Format the output in JSON format as follows, just give the JSON as output, nothing before it:

        {{
            "summary": {{
                "topic1": "value1",
                "topic2": "value2",
                "topic3": "value3"
            }},
            "questions": {{
                "{difficulty}": [
                    {{
                        "question": "What is the capital of France?",
                        "options": ["Paris", "London", "Berlin", "Madrid"],
                        "answer": "Paris"
                    }},
                    {{
                        "question": "What is the capital of Germany?",
                        "options": ["Paris", "London", "Berlin", "Madrid"],
                        "answer": "Berlin"
                    }}
                ]
            }}
        }}

        Transcript: {transcript}
        """

        if model_type.lower() == 'chatgroq':
            groq_model = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                groq_api_key=Config.GROQ_API_KEY
            )
            response = await groq_model.ainvoke(prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
        else:
            genai.configure(api_key=Config.GENAI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            response = await gemini_model.generate_content_async(prompt)
            response_content = response.text if hasattr(response, 'text') else str(response)

        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}, Raw response: {response_content}")
                return None
        else:
            print(f"No valid JSON found in response: {response_content}")
            return None
    except Exception as e:
        print(f"Error in generate_summary_and_quiz: {str(e)}")
        return None

async def fetch_youtube_transcript(video_url):
    try:
        video_id = re.search(r'(?:v=|youtu\.be/)([^&\n?#]+)', video_url)
        if not video_id:
            return {"error": "Invalid YouTube URL"}
        video_id = video_id.group(1)
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
        return formatter.format_transcript(transcript_data)
    except Exception as e:
        return {"error": f"Error fetching transcript: {str(e)}"}

async def generate_mind_map(content):
    prompt = f"""
    Extract key concepts from the following text and structure them into a JSON-based mind map.
    Organize it into: "Topic" -> "Subtopics" -> "Details".

    Text: {content}

    Output **ONLY** valid JSON in this format (no extra text, no explanations):
    {{
        "topic": "Main Topic",
        "subtopics": [
            {{"name": "Subtopic 1", "details": ["Detail 1", "Detail 2"]}},
            {{"name": "Subtopic 2", "details": ["Detail 3", "Detail 4"]}}
        ]
    }}
    """

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=Config.GROQ_API_KEY
    )

    response = await llm.ainvoke(prompt)
    raw_json = response.content.strip() if hasattr(response, 'content') else str(response)
    cleaned_json_str = raw_json.replace("```json", "").replace("```", "").replace("\n", "").strip()

    try:
        return json.loads(cleaned_json_str)
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON response: {cleaned_json_str}"}