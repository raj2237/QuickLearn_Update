from flask import request, jsonify
from functools import wraps
import jwt
from config import Config

def validate_token_middleware():
    def middleware(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            auth_header = request.headers.get("Authorization")
            token = auth_header.split("Bearer ")[-1] if auth_header and "Bearer " in auth_header else None
            
            if not token:
                return jsonify({"message": "Unauthorized: No token provided"}), 401
            
            try:
                decoded = jwt.decode(token, Config.SECRET_KEY, algorithms=["HS256"])
                request.user_id = decoded.get("id")
                request.user_role = decoded.get("role")
                return await func(*args, **kwargs)
            except jwt.ExpiredSignatureError:
                return jsonify({"message": "Unauthorized: Token has expired"}), 401
            except jwt.InvalidTokenError as e:
                print(f"Token decoding error: {e}")
                return jsonify({"message": "Unauthorized: Invalid token"}), 401
        
        return wrapper
    return middleware