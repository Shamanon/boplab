# config.py
import os

# Host & API Endpoints
WHISPER_URL = "http://localhost:30900/v1/audio/transcriptions"
OLLAMA_URL = "http://localhost:31434/api/generate"
MEATSPACE_API = "http://192.168.0.35:5000"  # Update onthewall IP here if needed

# Local File Paths
DB_PATH = "db/brain.db"
WAKE_MODELS = [
    "./models/hey_babs.onnx",
    "./models/bebopalula.onnx",
]
PIPER_MODEL = "./models/piper/en_US-amy-medium.onnx"

# Hardware Audio Settings
DEVICE_INPUT = 30  # PortAudio ALSA / Pulse device index for microphone
DEVICE_OUTPUT = 28
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280

# Dynamic VAD & Audio Thresholds
SILENCE_THRESHOLD = 2500
SILENCE_DURATION = 1.0
MAX_RECORD_SECONDS = 15
WAKE_THRESHOLD = 0.5

# Models
OLLAMA_MODEL = "babs-cyberpunk" #"llama3.2:3b"
MODEL_CHAT = "babs-cyberpunk"
MODEL_CODER = "babs-cyberpunk" #"qwen2.5-coder"

# Common Whisper hallucinations
WHISPER_HALLUCINATIONS = [
    "thank you.",
    "thank you",
    "thank you!",
    "thanks for watching.",
    "thanks for watching!",
    "thank you for watching.",
    "thank you for watching!",
    "subtitles by",
    "amara.org",
    "you",
    "bye.",
    "bye",
    "ご視聴ありがとうございました",
]
