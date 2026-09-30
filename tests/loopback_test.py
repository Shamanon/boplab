import io
import os
import sys
import wave
import subprocess
import numpy as np
import openwakeword
from openwakeword.model import Model

# Ensure ALSA driver stability
os.environ["PA_ALSA_DISABLE_PULSEAUDIO"] = "0"
os.environ["PA_ALSA_DISABLE_JACK"] = "1"
import sounddevice as sd

# Configuration
DEVICE_NUMBER = 30
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
MODEL_PATH = "./models/hey_babs.onnx"

print("[*] Loading Hey Babs Wake-Word Model...", flush=True)
if not os.path.exists(MODEL_PATH):
    print(f"[!] Error: Model file not found at {MODEL_PATH}", flush=True)
    sys.exit(1)

wake_model = Model(wakeword_model_paths=[MODEL_PATH])


def play_quick_beep():
    """Generates a quick, crisp 50ms 800Hz beep directly via raw PCM audio."""
    duration = 0.05  # 50 milliseconds
    freq = 800.0  # 800 Hz pitch
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    # Generate 16-bit PCM sine wave
    sine_wave = (np.sin(2 * np.pi * freq * t) * 16384).astype(np.int16)

    # Stream raw audio straight to aplay without external tool delays
    aplay_cmd = f"aplay -r {SAMPLE_RATE} -f S16_LE -c 1 -q -"
    proc = subprocess.Popen(aplay_cmd, shell=True, stdin=subprocess.PIPE)
    proc.communicate(input=sine_wave.tobytes())


def playback_recorded_audio(pcm_data):
    """Plays back the captured PCM buffer out of speakers via aplay."""
    print("[🔊] PLAYBACK: Outputting recorded audio to speakers...", flush=True)

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data.tobytes())

    wav_buffer.seek(0)

    # Pipe WAV directly to aplay
    proc = subprocess.Popen("aplay -q -", shell=True, stdin=subprocess.PIPE)
    proc.communicate(input=wav_buffer.read())


def main():
    print("\n==================================================", flush=True)
    print(" BABS Audio Loopback Test", flush=True)
    print(" 1. Say 'Hey Babs'", flush=True)
    print(" 2. Listen for the quick BEEP", flush=True)
    print(" 3. Speak immediately for 5 seconds", flush=True)
    print(" 4. Listen to the recording playback", flush=True)
    print("==================================================\n", flush=True)

    state = "LISTENING_WAKEWORD"
    command_buffer = []
    frames_needed = int((SAMPLE_RATE / CHUNK_SIZE) * 5)  # Exactly 5 seconds

    def audio_callback(indata, frames, time_info, status):
        nonlocal state, command_buffer

        # Audio chunk from microphone
        audio_chunk = (indata[:, 0] * 32767).astype(np.int16)

        if state == "LISTENING_WAKEWORD":
            wake_model.predict(audio_chunk)
            for model_name, scores in wake_model.prediction_buffer.items():
                if scores[-1] > 0.5:
                    wake_model.reset()
                    state = "WAKE_DETECTED"

        elif state == "RECORDING_COMMAND":
            command_buffer.append(audio_chunk)
            if len(command_buffer) >= frames_needed:
                state = "PROCESSING"

    # Single continuous stream on DEVICE_NUMBER
    with sd.InputStream(
        device=DEVICE_NUMBER,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        callback=audio_callback,
    ):
        print(f"[*] Stream active on input device {DEVICE_NUMBER}.", flush=True)

        while True:
            if state == "WAKE_DETECTED":
                print("\n[!] WAKE WORD DETECTED!", flush=True)

                # 1. Play clean 50ms beep
                play_quick_beep()

                # 2. Reset buffer & start capturing immediately
                command_buffer = []
                state = "RECORDING_COMMAND"
                print(
                    "[🔴] RECORDING... (Speak NOW for 5 seconds)", flush=True
                )

            elif state == "PROCESSING":
                print("[*] 5 seconds captured. Preparing playback...", flush=True)

                # Concatenate the recorded audio chunks
                full_pcm = np.concatenate(command_buffer, axis=0)

                # Check audio volume level / RMS
                rms = np.sqrt(np.mean(full_pcm.astype(np.float32) ** 2))
                print(f"[*] Audio Energy Level (RMS): {rms:.2f}", flush=True)

                if rms < 50:
                    print(
                        "[⚠️] WARNING: Audio level is near zero! Check mic volume or gain.",
                        flush=True,
                    )

                # Play back captured audio
                playback_recorded_audio(full_pcm)

                # Reset state back to wake word detection
                wake_model.reset()
                command_buffer = []
                state = "LISTENING_WAKEWORD"
                print("\n[*] Resetting... Say 'Hey Babs' to try again.", flush=True)

            sd.sleep(50)


if __name__ == "__main__":
    main()
