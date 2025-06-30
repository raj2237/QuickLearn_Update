from flask import Blueprint, request, jsonify
from utils.quiz_utils import generate_quiz
from utils.transcript_utils import get_and_enhance_transcript, generate_summary_and_quiz
import json 
quiz_bp = Blueprint('quiz', __name__)

@quiz_bp.route('/quiz', methods=['POST', 'OPTIONS'])
async def quiz():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    data = request.json
    youtube_link = data.get('link')
    num_questions = int(data.get('qno', 5))
    difficulty = data.get('difficulty', 'medium')
    model_type = data.get('model', 'chatgroq')

    if not youtube_link:
        return jsonify({"error": "No YouTube URL provided"}), 400

    transcript, language = await get_and_enhance_transcript(youtube_link, model_type)
    if not transcript:
        return jsonify({"error": "Failed to fetch transcript"}), 404

    summary_and_quiz = await generate_summary_and_quiz(transcript, num_questions, language, difficulty, model_type)
    if summary_and_quiz:
        return jsonify(summary_and_quiz)
    else:
        return jsonify({"error": "Failed to generate quiz"}), 500

@quiz_bp.route("/llm_quiz", methods=["POST"])
async def quiz_endpoint():
    data = request.json
    topic = data.get("topic")
    num_questions = data.get("num_questions")
    difficulty = data.get("difficulty")
    
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    
    try:
        response_content = await generate_quiz(topic, num_questions, difficulty)
        try:
            result = json.loads(response_content)
        except json.JSONDecodeError:
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response_content[json_start:json_end]
                result = json.loads(json_str)
            else:
                return jsonify({"error": "Could not parse JSON from response"}), 500
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500