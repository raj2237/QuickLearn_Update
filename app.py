from flask import Flask
from flask_cors import CORS
from config import Config
from routes.quiz_routes import quiz_bp
from routes.transcript_routes import transcript_bp
from routes.file_routes import file_bp
from routes.recommendation_routes import recommendation_bp
from routes.question_bank_routes import question_bank_bp
from asgiref.wsgi import WsgiToAsgi

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize CORS
    CORS(app, resources={
        r"/*": {
            "origins": [
                "http://localhost:5173",
                "http://localhost:3000",
                "http://localhost:3001"
            ],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        }
    })

    # Register blueprints
    app.register_blueprint(quiz_bp, url_prefix='/api')
    app.register_blueprint(transcript_bp, url_prefix='/api')
    app.register_blueprint(file_bp, url_prefix='/api')
    app.register_blueprint(recommendation_bp, url_prefix='/api')
    app.register_blueprint(question_bank_bp, url_prefix='/api')

    @app.route('/', methods=['GET'])
    async def health():
        return {"status": "ok"}

    return app

app = create_app()
asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:asgi_app", host="0.0.0.0", port=5001, workers=4)