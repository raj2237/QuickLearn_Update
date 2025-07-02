from flask import Flask
from flask_cors import CORS
from config import Config
from routes.quiz_routes import quiz_bp
from routes.transcript_routes import transcript_bp
from routes.file_routes import file_bp
from routes.recommendation_routes import recommendation_bp
from routes.question_bank_routes import question_bank_bp
from asgiref.wsgi import WsgiToAsgi
import os

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
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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

    # Remove async from Flask route - Flask doesn't support async routes natively
    @app.route('/', methods=['GET'])
    def health():
        return {"status": "ok", "message": "Flask app is running"}

    # Add error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {"error": "Not found"}, 404

    @app.errorhandler(500)
    def internal_error(error):
        return {"error": "Internal server error"}, 500

    return app

app = create_app()
asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    # Check if we're in development mode
    debug_mode = os.getenv('FLASK_ENV') == 'development' or os.getenv('DEBUG', 'False').lower() == 'true'
    
    if debug_mode:
        # Use Flask's built-in development server with auto-reload
        print("Running in development mode with Flask dev server...")
        app.run(
            host="0.0.0.0", 
            port=5001, 
            debug=True,  # Enables auto-reload and better error messages
            threaded=True
        )
    else:
        # Use Uvicorn for production
        import uvicorn
        print("Running in production mode with Uvicorn...")
        uvicorn.run(
            "app:asgi_app", 
            host="0.0.0.0", 
            port=5001, 
            workers=4,
            reload=True # Disable reload in production
        )