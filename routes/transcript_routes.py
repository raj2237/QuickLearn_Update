from flask import Blueprint, request, jsonify
from utils.transcript_utils import get_and_enhance_transcript, fetch_youtube_transcript, generate_mind_map
from langchain.prompts import PromptTemplate
from config import Config
from langchain_groq import ChatGroq
import google.generativeai as genai
transcript_bp = Blueprint('transcript', __name__)
from google.protobuf.json_format import MessageToDict

@transcript_bp.route('/chat_trans', methods=['POST', 'OPTIONS'])
async def chat_with_transcript():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        youtube_link = data.get('link')
        model_type = data.get('model', 'gemini')
        question = data.get('question')

        if not youtube_link:
            return jsonify({'error': 'Missing YouTube link'}), 400

        transcript, language = await get_and_enhance_transcript(youtube_link, model_type)
        
        if "Error" in transcript:
            return jsonify({'error': transcript}), 400

        if not question:
            return jsonify({
                'transcript': transcript,
                'language': language,
                'status': 'success'
            })

        # groq_model = ChatGroq(
        #     model="llama-3.3-70b-versatile",
        #     temperature=0,
        #     groq_api_key=Config.GROQ_API_KEY
        # )

        genai.configure(api_key=Config.GENAI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        

        prompt_template = PromptTemplate(
            input_variables=["transcript", "question"],
            template="""Given the following YouTube video transcript:
            {transcript}
            
            Please answer this question based on the transcript content:
            {question}"""
        )

        formatted_prompt = prompt_template.format(
            transcript=transcript,
            question=question
        )

        response = await gemini_model.generate_content_async(formatted_prompt)

# Get the first candidate's text directly
        generated_text = response.candidates[0].content.parts[0].text

        # Clean unwanted formatting
        cleaned_text = generated_text.replace('**', '').replace('\n', ' ').strip()

        return jsonify({
            'answer': cleaned_text,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@transcript_bp.route("/generate_mind_map", methods=['POST'])
async def generate_mind_map_endpoint():
    video_url = request.json.get('video_url')
    # print(video_url)
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400

    transcript = await fetch_youtube_transcript(video_url)
    # print(transcript)
    # if transcript:
    #     return jsonify(transcript), 400

    mind_map = await generate_mind_map(transcript)
    
    return jsonify(mind_map)