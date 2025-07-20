import aiohttp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from langchain_groq import ChatGroq
import google.generativeai as genai
import re
import json
from config import Config
import asyncio
import os
formatter = TextFormatter()
def format_transcript(transcript_data):
    return "\n".join([entry['text'].strip() for entry in transcript_data if entry['text'].strip()])

async def get_and_enhance_transcript(youtube_url, model_type='gemini'):
    # print(Config.GROQ_API_KEY)
    api = os.getenv("GROQ_API_KEY")
    # print(api)
    try:
        video_id_match = re.search(r'(?:v=|youtu\.be/)([^&\n?#]+)', youtube_url)
        if not video_id_match:
            print(f"Invalid YouTube URL: {youtube_url}")
            return None, None
        video_id = video_id_match.group(1)

        transcript_data = None
        language = None

        # Try fetching manually created or auto-generated transcript in English or Hindi
        for lang in ['en', 'hi']:
            try:
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                # print(transcript_data)
                language = lang
                # print(language)
                break
            except Exception:
                continue

        # Try translated transcript if direct ones fail
        if not transcript_data:
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                for transcript in transcript_list:
                    if transcript.is_translatable:
                        if 'en' in [t.language_code for t in transcript_list.translation_languages]:
                            transcript_data = transcript.translate('en').fetch()
                            language = 'hi'  # original was likely hindi
                            break
            except Exception as e:
                print(f"Transcript fetch failed: {e}")
                return None, None

        if not transcript_data:
            print(f"No transcript available for video ID: {video_id}")
            return None, None

        # Format transcript text
        formatted_transcript = format_transcript(transcript_data)

        # Prepare prompt
        prompt = f"""
        Act as a transcript cleaner. Generate a clean transcript based on the following:
        - Keep context and content same
        - Revise poorly structured parts
        - Output line by line
        - If the transcript lacks educational content, return "Fake transcript"
        Transcript:
        {formatted_transcript}
        """

        # Gemini
        if model_type.lower() == 'gemini':
            genai.configure(api_key=Config.GENAI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            response = await gemini_model.generate_content_async(prompt)
            enhanced_transcript = response.text.strip()
        # ChatGroq
        elif model_type.lower() == 'chatgroq':
            groq_model = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                groq_api_key=os.getenv("GROQ_API_KEY")
             )
            response = await groq_model.ainvoke(prompt)
            enhanced_transcript = response.content.strip() if hasattr(response, 'content') else str(response)
        else:
            print("Invalid model type selected.")
            return None, None

        return enhanced_transcript, language

    except Exception as e:
        print(f"Error in get_and_enhance_transcript: {e}")
        return None, None

async def generate_summary_and_quiz(transcript, num_questions, language, difficulty, model_type='gemini'):
    try:
        if 'Fake transcript' in transcript:
            return {"summary": {}, "questions": {difficulty: []}}

        prompt = f"""
        Analyze the following transcript and create a comprehensive structured summary along with a quiz.
        
        For the summary, organize it into a clear hierarchical structure with:
        1. Main topic identification (2-3 words)
        2. Key concepts with their explanations in about 3-5 detailed lines which users can relate technically and understand
        3. Important points under each concept
        4. Practical applications or examples mentioned, if the examples are not mentioned , generate 2-3 concept related examples.
        
        Structure the output in the following JSON format, ensuring all text uses proper apostrophes (') for possessives and contractions:

        {{
            "summary": {{
                "main_topic": "Brief topic name (2-3 words)",
                "overview": "2-3 sentence overview of the entire content",
                "key_concepts": [
                    {{
                        "concept": "Concept Name",
                        "explanation": "Clear explanation of this concept",
                        "key_points": [
                            "Important point 1",
                            "Important point 2",
                            "Important point 3"
                        ],
                        "examples": [
                            "Example or application if mentioned"
                        ]
                    }}
                ],
                "takeaways": [
                    "Main takeaway 1",
                    "Main takeaway 2",
                    "Main takeaway 3"
                ]
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

        Generate exactly {num_questions} multiple-choice questions at {difficulty} difficulty level based on the transcript content.
        Ensure all content is in English regardless of the original transcript language.
        If the transcript contains 'Fake Transcript', return empty summary and questions.
        
        Output ONLY the JSON, nothing else.

        Transcript: {transcript}
        """

        if model_type.lower() == 'chatgroq':
            groq_model = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                groq_api_key=os.getenv("GROQ_API_KEY")
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
        # print(transcript_data)
        return transcript_data
        
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

    genai.configure(api_key=Config.GENAI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    response = await gemini_model.generate_content_async(prompt)

    response_content = response.text if hasattr(response, 'text') else str(response)
    cleaned_json_str = response_content.replace("```json", "").replace("```", "").replace("\n", "").strip()

    try:
        return json.loads(cleaned_json_str)
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON response: {cleaned_json_str}"}

# Additional utility function to format the structured summary for display
def format_summary_for_display(summary_data):
    """
    Convert the structured summary JSON into a more readable format
    """
    if not summary_data or 'summary' not in summary_data:
        return "No summary available"
    
    summary = summary_data['summary']
    formatted_output = []
    
    # Main Topic and Overview
    formatted_output.append(f"📚 **Topic:** {summary.get('main_topic', 'Unknown')}")
    formatted_output.append(f"📝 **Overview:** {summary.get('overview', 'No overview available')}")
    formatted_output.append("")
    
    # Key Concepts
    if 'key_concepts' in summary and summary['key_concepts']:
        formatted_output.append("🔍 **Key Concepts:**")
        formatted_output.append("")
        
        for i, concept in enumerate(summary['key_concepts'], 1):
            formatted_output.append(f"**{i}. {concept.get('concept', 'Unnamed Concept')}**")
            formatted_output.append(f"   {concept.get('explanation', 'No explanation available')}")
            
            if concept.get('key_points'):
                formatted_output.append("   **Key Points:**")
                for point in concept['key_points']:
                    formatted_output.append(f"   • {point}")
            
            if concept.get('examples'):
                formatted_output.append("   **Examples:**")
                for example in concept['examples']:
                    formatted_output.append(f"   → {example}")
            
            formatted_output.append("")
    
    # Main Takeaways
    if 'takeaways' in summary and summary['takeaways']:
        formatted_output.append("💡 **Key Takeaways:**")
        for takeaway in summary['takeaways']:
            formatted_output.append(f"• {takeaway}")
    
    return "\n".join(formatted_output)