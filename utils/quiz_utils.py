from langchain_groq import ChatGroq
from config import Config
import json

def generate_quiz(topic: str, num_questions: int, difficulty: str):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=Config.GROQ_API_KEY
    )

    prompt = f"""
    Create a quiz on the topic: "{topic}". Generate {num_questions} multiple-choice questions.
    The questions should be of {difficulty} difficulty.
    Format the output strictly in JSON format as follows:
    
    {{
        "questions": {{
            "{difficulty}": [
                {{
                    "question": "What is ...?",
                    "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                    "answer": "Option 1"
                }}
            ]
        }}
    }}
    """
    response = llm.invoke(prompt)
    return response.content if hasattr(response, 'content') else response.text