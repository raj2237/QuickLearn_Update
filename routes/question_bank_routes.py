from flask import Blueprint, request, jsonify, send_file
from utils.pdf_utils import extract_questions_from_pdf, generate_questions, create_question_paper, create_question_bank_pdf
from config import Config
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
import os
import random
import aiofiles
import io
from contextlib import redirect_stdout
import unicodedata
import re

question_bank_bp = Blueprint('question_bank', __name__)

@question_bank_bp.route('/paper_upload', methods=['POST'])
async def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    file_path = os.path.join(Config.UPLOAD_FOLDER, file.filename)
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(file.read())
    
    return jsonify({"message": "File uploaded successfully", "file_path": file_path})

@question_bank_bp.route('/generate_paper', methods=['POST'])
async def generate_papers():
    data = request.json
    pdf_path = data.get("file_path")
    num_questions = data.get("num_questions", 10)
    num_papers = data.get("num_papers", 1)
    
    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({"error": "Invalid file path"}), 400
    
    try:
        extracted_questions = await extract_questions_from_pdf(pdf_path)
        if not extracted_questions:
            return jsonify({"error": "No valid questions found in PDF"}), 400
        
        generated_questions = await generate_questions(extracted_questions, num_questions * num_papers)
        all_questions = extracted_questions + generated_questions
        random.shuffle(all_questions)
        
        pdf_paths = []
        for i in range(num_papers):
            start_idx = i * num_questions
            end_idx = start_idx + num_questions
            if start_idx >= len(all_questions):
                break
                
            selected_questions = all_questions[start_idx:end_idx]
            if len(selected_questions) < num_questions:
                remaining = num_questions - len(selected_questions)
                extra_questions = await generate_questions(extracted_questions, remaining)
                selected_questions.extend(extra_questions)
            
            paper_path = os.path.join(Config.OUTPUT_FOLDER, f"question_paper_set_{i+1}.pdf")
            await create_question_paper(selected_questions, paper_path, i + 1)
            pdf_paths.append(paper_path)
        
        return jsonify({"message": "Papers generated", "files": pdf_paths})
    
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@question_bank_bp.route('/download/<filename>', methods=['GET'])
async def download_paper(filename):
    file_path = os.path.join(Config.OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return await send_file(file_path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

@question_bank_bp.route('/question_bank', methods=['POST', 'OPTIONS'])
async def generate_question_bank():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        subject = data.get('topic')
        
        if not subject or not isinstance(subject, str):
            return jsonify({"error": "Invalid or missing 'topic' in payload"}), 400
        
        subject = subject.strip()
        if not subject:
            return jsonify({"error": "Topic cannot be empty"}), 400
        
        agent = initialize_question_bank_agent()
        
        prompt = (
            f"Create 10 challenging practice problems on {subject}. "
            "Number each question with a number and period (1., 2., etc.). "
            "Make each question clear and well-formatted on its own line. "
            "Include a mix of difficulty levels from basic to advanced. "
            "Focus on numerical questions over theoretical ones."
        )
 
        result_text = await get_agent_response(agent, prompt)
        
        if result_text.startswith("1. Error:") or result_text.startswith("1. An error occurred"):
            return jsonify({"error": "Failed to generate questions"}), 500
        
        pdf_path = await create_question_bank_pdf(result_text, subject)
        
        return await send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"{subject.replace(' ', '_')}_Questions.pdf",
            mimetype='application/pdf'
        )
    
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def initialize_question_bank_agent():
    try:
        groq_model = Groq(
            id="llama3-70b-8192",
            api_key=Config.GROQ_API_KEY
        )
        agent = Agent(
            model=groq_model,
            description=(
                "You are a helpful assistant that creates challenging practice problems. "
                "When asked to create questions, format them with numbers (1., 2., etc.) "
                "and make them clear and well-structured."
            ),
            tools=[DuckDuckGoTools()],
            show_tool_calls=False,
            markdown=False
        )
        return agent
    except Exception as e:
        raise Exception(f"Error initializing agent: {str(e)}")

async def get_agent_response(agent, prompt):
    try:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            await agent.aprint_response(prompt)
        response = buffer.getvalue()
        
        if not response:
            raise Exception("No response received from agent")
        
        cleaned = unicodedata.normalize('NFKD', response).encode('ascii', 'ignore').decode('ascii')
        cleaned = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', cleaned)
        cleaned = re.sub(r'Running:.*?\n', '', cleaned)
        cleaned = re.sub(r'Response:?\s*', '', cleaned)
        cleaned = re.sub(r'<tool-use>.*?</tool-use>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\(\d+\.\d+s\)', '', cleaned)
        cleaned = re.sub(r'Here are \d+ practice problems.*?:(?=\n)', '', cleaned)
        cleaned = re.sub(r'1\., 2\., etc\.\).*?\n', '', cleaned)
        cleaned = re.sub(r'Make each question clear and well-formatted.*?numerical questions over theory\.', '', cleaned)
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        questions = []
        current_question = ""
        for line in cleaned.split('\n'):
            line = line.strip()
            if not line:
                continue
            if re.match(r'^\d+\.', line):
                if current_question:
                    questions.append(current_question.strip())
                current_question = line
            elif current_question:
                current_question += " " + line
        
        if current_question:
            questions.append(current_question.strip())
        
        if not questions:
            matches = re.findall(r'(\d+\..*?)(?=\d+\.|$)', cleaned, re.DOTALL)
            questions = [q.strip() for q in matches]
        
        if not questions:
            return "1. Error: No questions could be generated. Please try again."
        
        formatted_questions = []
        for i, q in enumerate(questions, 1):
            q = q.strip()
            if not re.match(r'^\d+\.', q):
                q = f"{i}. {q}"
            formatted_questions.append(q)
        
        return "\n\n".join(formatted_questions)
        
    except Exception as e:
        return f"1. An error occurred while generating questions: {str(e)}"