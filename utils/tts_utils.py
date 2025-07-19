import threading
import logging
from bs4 import BeautifulSoup
import html
import re
import requests
import base64
import os
import pygame
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextToSpeechManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.api_key = Config.SARVAM_API_KEY
        self.api_url = "https://api.sarvam.ai/text-to-speech"
        pygame.mixer.init()  # Initialize pygame mixer for audio playback

    def speak(self, text):
        try:
            with self.lock:
                # Prepare payload for Sarvam TTS API
                payload = {
                    "inputs": [text[:500]],  # Limit to 500 characters per API docs
                    "target_language_code": "en-IN",  # Default to English-India
                    "speaker": "anushka",  # Default speaker, customizable
                    "pitch": 0.15,  # Range: -0.75 to 0.75
                    "pace": 0.9,  # Range: 0.5 to 2.0, faster for liveliness
                    "loudness": 1.55,  # Range: 0.3 to 3.0
                    "speech_sample_rate": 8000,  # Supported: 8000, 16000, 22050, 24000 Hz
                    "enable_preprocessing": False,
                    "model": "bulbul:v2"  # Latest model
                }
                headers = {
                    "API-Subscription-Key": self.api_key,
                    "Content-Type": "application/json"
                }

                # Make API request
                response = requests.post(self.api_url, json=payload, headers=headers)
                if response.status_code != 200:
                    logger.error(f"Sarvam TTS API error: {response.status_code}, {response.text}")
                    return False

                # Decode base64 audio
                audio_data = response.json().get("audios", [])[0]  # First audio in response
                audio_bytes = base64.b64decode(audio_data)

                # Save to temporary file
                temp_file = "temp_sarvam_audio.wav"
                with open(temp_file, "wb") as f:
                    f.write(audio_bytes)

                # Play audio using pygame
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():  # Wait for playback to finish
                    pygame.time.Clock().tick(10)

                # Clean up
                pygame.mixer.music.unload()
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                logger.info("Speech completed successfully")
                return True
        except Exception as e:
            logger.error(f"Text-to-speech error: {str(e)}")
            return False

    def start_speaking(self, text):
        thread = threading.Thread(target=self.speak, args=(text,))
        thread.daemon = True
        thread.start()
        return thread

tts_manager = TextToSpeechManager()

def clean_response(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = html.unescape(text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = ' '.join(text.split())
    return text