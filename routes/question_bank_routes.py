from flask import Blueprint, request, jsonify, send_file
from utils.pdf_utils import extract_questions_from_pdf, generate_questions, create_question_paper, create_question_bank_pdf
from config import Config
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
import os
import random
import io
from contextlib import redirect_stdout
import unicodedata
import re
from datetime import datetime
import asyncio

question_bank_bp = Blueprint('question_bank', __name__)

@question_bank_bp.route('/paper_upload', methods=['POST'])
def upload_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "File must be a PDF"}), 400
        
        # Create a unique filename using timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"uploaded_paper_{timestamp}.pdf"
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        
        # Save file synchronously
        file.save(file_path)
        
        return jsonify({"message": "File uploaded successfully", "file_path": file_path}), 200
    
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@question_bank_bp.route('/generate_paper', methods=['POST'])
def generate_papers():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        pdf_path = data.get("file_path")
        num_questions = data.get("num_questions", 10)
        num_papers = data.get("num_papers", 1)
        
        if not pdf_path or not os.path.exists(pdf_path):
            return jsonify({"error": "Invalid or missing file path"}), 400
        
        if not isinstance(num_questions, int) or num_questions < 1:
            return jsonify({"error": "Number of questions must be a positive integer"}), 400
        if not isinstance(num_papers, int) or num_papers < 1:
            return jsonify({"error": "Number of papers must be a positive integer"}), 400
        
        # Run async functions in event loop
        extracted_questions = asyncio.run(extract_questions_from_pdf(pdf_path))
        if not extracted_questions:
            return jsonify({"error": "No valid questions found in PDF"}), 400
        
        generated_questions = asyncio.run(generate_questions(extracted_questions, num_questions * num_papers))
        all_questions = extracted_questions + generated_questions
        random.shuffle(all_questions)
        
        # Create folder for generated papers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        papers_folder = os.path.join(Config.OUTPUT_FOLDER, f"question_papers_{timestamp}")
        os.makedirs(papers_folder, exist_ok=True)
        
        generated_files = []
        
        for i in range(num_papers):
            start_idx = i * num_questions
            end_idx = start_idx + num_questions
            if start_idx >= len(all_questions):
                break
            
            selected_questions = all_questions[start_idx:end_idx]
            if len(selected_questions) < num_questions:
                remaining = num_questions - len(selected_questions)
                extra_questions = asyncio.run(generate_questions(extracted_questions, remaining))
                selected_questions.extend(extra_questions)
            
            paper_filename = f"question_paper_set_{i+1}.pdf"
            paper_path = os.path.join(papers_folder, paper_filename)
            asyncio.run(create_question_paper(selected_questions, paper_path, i + 1))
            
            if os.path.exists(paper_path):
                generated_files.append({
                    "filename": paper_filename,
                    "path": paper_path,
                    "set_number": i + 1
                })
        
        if not generated_files:
            return jsonify({"error": "No question papers were generated"}), 500
        
        return jsonify({
            "message": f"Successfully generated {len(generated_files)} question paper(s)",
            "papers_folder": papers_folder,
            "generated_files": generated_files,
            "total_papers": len(generated_files),
            "questions_per_paper": num_questions
        }), 200
    
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@question_bank_bp.route('/question_bank', methods=['POST', 'OPTIONS'])
def generate_question_bank():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200
        
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
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
 
        result_text = get_agent_response_sync(agent, prompt)
        
        if result_text.startswith("1. Error:") or result_text.startswith("1. An error occurred"):
            return jsonify({"error": "Failed to generate questions"}), 500
        
        pdf_path = asyncio.run(create_question_bank_pdf(result_text, subject))
        
        return send_file(
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

def get_agent_response_sync(agent, prompt):
    """Synchronous wrapper for agent response"""
    try:
        # Try different methods to get response from agent
        if hasattr(agent, 'run'):
            # Method 1: Use run method if available
            response = agent.run(prompt)
        elif hasattr(agent, 'get_response'):
            # Method 2: Use get_response method if available
            response = agent.get_response(prompt)
        elif hasattr(agent, 'print_response'):
            # Method 3: Capture print_response output
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                agent.print_response(prompt)
            response = buffer.getvalue()
        else:
            # Method 4: Try to run async method in sync context
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(get_agent_response_async(agent, prompt))
                loop.close()
            except Exception as async_error:
                raise Exception(f"Could not get response from agent: {str(async_error)}")
        
        if not response:
            raise Exception("No response received from agent")
        
        # Convert response to string if it's not already
        if hasattr(response, 'content'):
            response_text = response.content
        elif hasattr(response, 'text'):
            response_text = response.text
        else:
            response_text = str(response)
        
        return clean_and_format_response(response_text)
    
    except Exception as e:
        return f"1. An error occurred while generating questions: {str(e)}"

async def get_agent_response_async(agent, prompt):
    """Async method to get agent response"""
    try:
        # Try different async methods
        if hasattr(agent, 'arun'):
            response = await agent.arun(prompt)
        elif hasattr(agent, 'aget_response'):
            response = await agent.aget_response(prompt)
        elif hasattr(agent, 'aprint_response'):
            # Capture async print response
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                result = agent.aprint_response(prompt)
                # Check if result is awaitable
                if hasattr(result, '__await__'):
                    await result
                else:
                    # If not awaitable, just execute it
                    pass
            response = buffer.getvalue()
        else:
            raise Exception("No suitable async method found on agent")
        
        return response
        
    except Exception as e:
        raise Exception(f"Async agent response failed: {str(e)}")

def clean_and_format_response(response_text):
    """Clean and format the agent response"""
    try:
        cleaned = unicodedata.normalize('NFKD', response_text).encode('ascii', 'ignore').decode('ascii')
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
        return f"1. An error occurred while formatting questions: {str(e)}"