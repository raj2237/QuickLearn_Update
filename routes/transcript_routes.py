from flask import Blueprint, request, jsonify
from utils.transcript_utils import get_and_enhance_transcript, fetch_youtube_transcript, generate_mind_map
from langchain.prompts import PromptTemplate
from config import Config
from langchain_groq import ChatGroq

transcript_bp = Blueprint('transcript', __name__)

@transcript_bp.route('/chat_trans', methods=['POST', 'OPTIONS'])
def chat_with_transcript():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        youtube_link = data.get('link')
        model_type = data.get('model', 'chatgroq')
        question = data.get('question')

        if not youtube_link:
            return jsonify({'error': 'Missing YouTube link'}), 400

        transcript, language = get_and_enhance_transcript(youtube_link, model_type)
        
        if "Error" in transcript:
            return jsonify({'error': transcript}), 400

        if not question:
            return jsonify({
                'transcript': transcript,
                'language': language,
                'status': 'success'
            })

        groq_model = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=Config.GROQ_API_KEY
        )

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

        response = groq_model.invoke(formatted_prompt)

        return jsonify({
            'answer': response.content,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@transcript_bp.route("/generate_mind_map", methods=['GET'])
def generate_mind_map_endpoint():
    video_url = request.args.get('video_url')

    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400

    transcript = fetch_youtube_transcript(video_url)
    if isinstance(transcript, dict) and "error" in transcript:
        return jsonify(transcript), 400

    mind_map = generate_mind_map(transcript)
   
    return jsonify(mind_map)