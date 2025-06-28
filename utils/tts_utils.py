import threading
import pyttsx3
import logging
from bs4 import BeautifulSoup
import html
import re
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextToSpeechManager:
    def __init__(self):
        self.lock = threading.Lock()
    
    def speak(self, text):
        try:
            with self.lock:
                engine = None
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.setProperty('volume', 1.0)
                    engine.say(text)
                    engine.runAndWait()
                    engine.startLoop(False)
                    engine.iterate()
                    engine.endLoop()
                    logger.info("Speech completed successfully")
                finally:
                    if engine:
                        try:
                            engine.stop()
                        except:
                            pass
                        del engine
        except Exception as e:
            logger.error(f"Text-to-speech error: {str(e)}")
            
    def start_speaking(self, text):
        thread = Thread(target=self.speak, args=(text,))
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