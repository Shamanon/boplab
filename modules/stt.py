# modules/stt.py
import requests
import config

def transcribe_audio(wav_buffer):
    """Pipes WAV buffer through local Whisper STT engine with hallucination filtering."""
    if not wav_buffer:
        return ""

    files = {"file": ("speech.wav", wav_buffer, "audio/wav")}
    data = {"model": "Systran/faster-whisper-small.en", "language": "en"}

    try:
        response = requests.post(
            config.WHISPER_URL, files=files, data=data, timeout=8
        )
        if response.status_code == 200:
            text = response.json().get("text", "").strip()
            
            # Filter out silent noise hallucinations
            if text.lower() in config.WHISPER_HALLUCINATIONS or len(text) < 2:
                return ""
                
            print(f'[🎤] Transcribed: "{text}"', flush=True)
            return text
    except Exception as e:
        print(f"[!] STT Error: {e}")
    return ""
