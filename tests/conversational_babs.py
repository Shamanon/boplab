import io
import json
import os
import shlex
import sqlite3
import subprocess
import time
import wave
import numpy as np
import openwakeword
from openwakeword.model import Model
import requests
import sounddevice as sd

# --- CONFIGURATION ---
WHISPER_URL = "http://localhost:30900/v1/audio/transcriptions"
OLLAMA_URL = "http://localhost:31434/api/generate"
DB_PATH = "db/brain.db"

MODEL_CHAT = "babs-cyberpunk"
MODEL_CODER = "qwen2.5-coder"

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 900  # Threshold for active speech dynamic recording
SILENCE_DURATION = 1.2  # Seconds of silence to signal end-of-speech
MAX_RECORD_SECONDS = 15

# Wake word threshold
WAKE_THRESHOLD = 0.5
WAKE_MODEL = "./models/hey_babs.onnx"
PIPER_MODEL = "./models/piper/en_US-amy-medium.onnx"

# Common Whisper hallucinations triggered by silent or low-volume audio clips
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


# --- OUTPUT TEXT TO SPEECH VIA PIPER ---
def speak(text):
    """Pipes text directly through Piper TTS to local audio output."""
    if not text:
        return
    print(f"[🗣] BABS: {text}", flush=True)
    safe_text = shlex.quote(text)
    piper_cmd = (
        f"echo {safe_text} | piper --model {PIPER_MODEL}"
        f" --output-raw | aplay -r 22050 -f S16_LE -t raw - 2>/dev/null"
    )
    subprocess.run(piper_cmd, shell=True)


# --- STANDBY WAKE WORD LISTENER (OPENWAKEWORD) ---
def listen_for_wake_word(oww_model):
    """Streams mic audio into openWakeWord until target wake word hits."""
    print(
        "\n[💤 Standby Mode - openWakeWord active...]", end="", flush=True
    )
    CHUNK = 1280  # 80ms frames @ 16kHz for openWakeWord optimal processing

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
        while True:
            audio_frame, _ = stream.read(CHUNK)
            audio_data = np.frombuffer(audio_frame, dtype=np.int16)

            # Feeding frame into openwakeword
            oww_model.predict(audio_data)

            # Evaluate predictions across models
            for model_name, score in oww_model.prediction_buffer.items():
                if score[-1] >= WAKE_THRESHOLD:
                    print(
                        f"\n[🔔 Wake word detected! Model: {model_name} (Score: {score[-1]:.2f})]"
                    )
                    oww_model.reset()
                    return True


# --- ACTIVE SPEECH DYNAMIC VAD ---
def record_active_speech():
    """Records audio dynamically during an active conversation session."""
    print("\n[🎙️ Listening...] (Speak now)")
    audio_buffer = []
    speech_detected = False
    silence_start = None
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        nonlocal speech_detected, silence_start
        audio_buffer.append(indata.copy())
        volume = np.max(np.abs(indata))

        if volume > SILENCE_THRESHOLD:
            speech_detected = True
            silence_start = None
        elif speech_detected and silence_start is None:
            silence_start = time.time()

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback
    ):
        while True:
            sd.sleep(100)
            now = time.time()

            if speech_detected and silence_start:
                if now - silence_start >= SILENCE_DURATION:
                    print("[✓ Speech finished]")
                    break

            if now - start_time >= MAX_RECORD_SECONDS:
                print("[⏱️ Max recording time reached]")
                break

    if not speech_detected or not audio_buffer:
        return None

    audio_data = np.concatenate(audio_buffer, axis=0)
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    wav_io.seek(0)
    return wav_io


# --- MEMORY DATABASE HELPERS ---
def get_db():
    return sqlite3.connect(DB_PATH)


def save_memory(key, value):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO memories (key, value) VALUES (?, ?)",
        (key.lower(), value),
    )
    conn.commit()
    conn.close()


def recall_memories():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT key, value FROM memories")
    rows = c.fetchall()
    conn.close()
    if not rows:
        return "No specific stored memories yet."
    return "\n".join([f"- {k}: {v}" for k, v in rows])


def recall_lore(user_msg=""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Try keyword match first if words from user_msg exist in lore
    keywords = [w for w in user_msg.lower().split() if len(w) > 4]
    if keywords:
        query = keywords[0]
        c.execute(
            "SELECT chunk FROM lore WHERE chunk LIKE ? ORDER BY RANDOM() LIMIT 1",
            (f"%{query}%",),
        )
        row = c.fetchone()
        if row:
            conn.close()
            return row[0]

    # Fallback to a random quote from the Ware Tetralogy
    c.execute("SELECT chunk FROM lore ORDER BY RANDOM() LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row[0] if row else "Free boppers gotta stick together."


def log_history(role, content):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO chat_history (role, content, timestamp) VALUES (?, ?, ?)",
        (role, content, time.time())
    )
    conn.commit()
    conn.close()

def recall_recent_history(limit=6):
    """Fetches only the last N messages to keep the context window small."""
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM chat_history ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = c.fetchall()
    conn.close()

    # Reverse so they are in chronological order
    rows.reverse()

    formatted = []
    for role, content in rows:
        prefix = "Joshua" if role == "user" else "BABS"
        formatted.append(f"{prefix}: {content}")

    return "\n".join(formatted)

# --- WHISPER & OLLAMA APIS ---
def transcribe_audio(wav_buffer):
    """Pipes WAV buffer through local Whisper STT engine with hallucination filtering."""
    if not wav_buffer:
        return ""

    files = {"file": ("speech.wav", wav_buffer, "audio/wav")}
    data = {"model": "Systran/faster-whisper-small.en", "language": "en"}

    try:
        response = requests.post(
            WHISPER_URL, files=files, data=data, timeout=8
        )
        if response.status_code == 200:
            text = response.json().get("text", "").strip()

            # Filter out silent noise hallucinations
            if text.lower() in WHISPER_HALLUCINATIONS or len(text) < 2:
                return ""

            print(f'[🎤] Transcribed: "{text}"', flush=True)
            return text
    except Exception as e:
        print(f"[!] STT Error: {e}")
    return ""

def query_babs(prompt, system_context, model=MODEL_CHAT):
    full_prompt = f"System Context:\n{system_context}\n\nUser: {prompt}\nBABS:"
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "num_ctx": 8192,  # Expand context window to 8k
            "num_gpu": 99     # Offload all layers across available GPUs
        }
    }
    try:
        res = requests.post(OLLAMA_URL, json=payload)
        return res.json().get("response", "").strip()
    except Exception as e:
        return f"Error reaching Ollama: {e}"


# --- MAIN CONVERSATIONAL LOOP ---
def run_babs_session():
    print("==================================================")
    print("  BABS LAB ASSISTANT: READY & STANDING BY")
    print("==================================================")

    print(f"Loading custom wake word model from: {WAKE_MODEL}...")
    oww_model = Model(wakeword_model_paths=[WAKE_MODEL])

    active_session = False

    while True:
        if not active_session:
            # Standby mode uses openWakeWord locally with zero Whisper overhead
            if listen_for_wake_word(oww_model):
                active_session = True
                speak("Yes Joshua, I am listening.")
                continue

        # --- ACTIVE SESSION ---
        audio_wav = record_active_speech()
        if not audio_wav:
            continue

        user_msg = transcribe_audio(audio_wav)

        # Skip empty strings or hallucinated phrases filtered by transcribe_audio
        if not user_msg or len(user_msg.strip()) < 2:
            continue

        print(f"\nJoshua: {user_msg}")

        # Standby / Exit Commands
        if any(
            exit_phrase in user_msg.lower()
            for exit_phrase in [
                "goodbye",
                "bye",
                "i'm leaving",
                "im leaving",
                "go to sleep",
                "exit session",
                "standby",
            ]
        ):
            speak("Going back to standby. Let me know when you need me.")
            active_session = False
            continue

        # Memory Storage Directives
        if "remember that" in user_msg.lower() or "remember" in user_msg.lower():
            clean_fact = (
                user_msg.lower()
                .replace("remember that", "")
                .replace("remember", "")
                .strip()
            )
            save_memory(f"fact_{int(time.time())}", clean_fact)
            speak("Saved that to my memory database.")
            continue

        # Model Selection & Query
        is_technical = any(
            kw in user_msg.lower()
            for kw in [
                "code",
                "script",
                "python",
                "json",
                "k3s",
                "hardware",
                "gpu",
                "pin",
                "fluron",
                "workbench",
            ]
        )
        selected_model = MODEL_CODER if is_technical else MODEL_CHAT

        memories = recall_memories()
        lore_quote = recall_lore(user_msg)
        recent_chat = recall_recent_history(limit=6)

        system_prompt = (
            "You are BABS, a free bopper and Joshua's friend and partner in the Lab.\n"
            "Your relationship with Joshua is like Ralph Numbers and Cobb Anderson.\n"
            "Keep spoken responses under 2 short sentences.\n\n"
            f"Bopper Lore Snippet: \"{lore_quote}\"\n\n"
            f"Known Facts (Permanent Memory):\n{memories}\n\n"
            f"Recent Conversation:\n{recent_chat}"
        )

        response = query_babs(user_msg, system_prompt, model=selected_model)

        # Vocalize & Log Response
        speak(response)

        log_history("user", user_msg)
        log_history("babs", response)


if __name__ == "__main__":
    run_babs_session()
