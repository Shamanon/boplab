import os
import sys
import numpy as np
import openwakeword
from openwakeword.model import Model
import subprocess

# Ensure clean ALSA/PortAudio initialization
os.environ["PA_ALSA_DISABLE_PULSEAUDIO"] = "0"
os.environ["PA_ALSA_DISABLE_JACK"] = "1"
import sounddevice as sd

# Configuration
DEVICE_NUMBER = 30
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
MODEL_PATH = "./models/hey_babs.onnx"

print("[*] Loading Hey Babs wake word model...")
if not os.path.exists(MODEL_PATH):
    print(f"[!] Error: Model file not found at {MODEL_PATH}")
    sys.exit(1)

wake_model = Model(wakeword_model_paths=[MODEL_PATH])
print("[✔] Model loaded successfully.")

def speak_response():
    """Outputs speech using piper or a direct fallback alert."""
    print("[🗣] Triggered! Responding: 'OK, I hear you'")
    
    # Try Piper TTS if available
    piper_cmd = (
        'echo "OK, I hear you" | piper --model ./models/piper/en_US-amy-medium.onnx'
        ' --output-raw | aplay -r 22050 -f S16_LE -t raw - 2>/dev/null'
    )
    res = subprocess.run(piper_cmd, shell=True)
    
    # Fallback to espeak or system beep if Piper fails or isn't built
    if res.returncode != 0:
        subprocess.run('espeak-ng "OK, I hear you" 2>/dev/null || speaker-test -t sine -f 1000 -l 1 -s 1 2>/dev/null', shell=True)

def main():
    print("\n==================================================")
    print(" BABS Wake Word Diagnostic Test")
    print(" Speak 'Hey Babs' into the microphone...")
    print("==================================================\n")

    triggered = False

    def audio_callback(indata, frames, time, status):
        nonlocal triggered
        if status:
            print(f"[!] SoundDevice Status: {status}")
            
        audio_chunk = (indata[:, 0] * 32767).astype(np.int16)
        wake_model.predict(audio_chunk)

        for model_name, scores in wake_model.prediction_buffer.items():
            if scores[-1] > 0.5:
                print(f"\n[!] WAKE WORD DETECTED ({model_name} score: {scores[-1]:.2f})")
                wake_model.reset()
                triggered = True

    while True:
        triggered = False
        
        # Open stream inside context manager
        with sd.InputStream(
            device=DEVICE_NUMBER,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
        ):
            print("[*] Listening...")
            while not triggered:
                sd.sleep(100)

        # Stream automatically closes here upon exit of 'with' block, freeing ALSA hardware
        print("[*] Microphones temporarily suspended for playback...")
        speak_response()
        sd.sleep(500)  # Brief pause before re-opening input stream

if __name__ == "__main__":
    main()
