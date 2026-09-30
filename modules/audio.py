# modules/audio.py
import os
import wave
import time
import io
import shlex
import subprocess
import numpy as np
import onnxruntime as ort
from piper import PiperVoice
import sounddevice as sd
import config
from openwakeword.model import Model

os.environ["PA_ALSA_DISABLE_PULSEAUDIO"] = "0"
os.environ["PA_ALSA_DISABLE_JACK"] = "1"
os.environ["ORT_LOGGING_LEVEL"] = "4"

# Initialize Piper Engine in Python with CUDA support
print("[*] Initializing Piper Voice Engine in Python...")
wake_model = Model(wakeword_model_paths=config.WAKE_MODELS)
print("[✔] Active Bopper Triggers:", list(wake_model.models.keys()))

# 1. Silence ONNX Runtime Verbose Warnings
session_options = ort.SessionOptions()
session_options.log_severity_level = 4  # 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal

try:
  voice = PiperVoice.load(
      config.PIPER_MODEL,
      config_path=f"{config.PIPER_MODEL}.json",
      use_cuda=True,
  )
  print("[✔] Piper loaded with CUDA acceleration.")
except Exception as e:
  print(
      f"[!] CUDA failed ({e}), falling back to CPU..."
  )
  voice = PiperVoice.load(config.PIPER_MODEL, use_cuda=False)

def babs_speak(text):
  """Synchronous low-latency speech synthesis via sounddevice."""
  if not text:
    return

  print(f"[🗣] BABS: {text}", flush=True)

  try:
    # Query default or selected hardware device channel requirements
    device_info = sd.query_devices(config.DEVICE_OUTPUT, "output")
    target_channels = min(2, device_info["max_output_channels"])

    with sd.OutputStream(
        samplerate=22050,
        channels=target_channels,
        dtype="int16",
        device=config.DEVICE_OUTPUT,
    ) as stream:
      for chunk in voice.synthesize(text):
        audio_data = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)

        # If device requires stereo (2 channels), duplicate mono channel across both left/right
        if target_channels == 2:
          audio_data = np.column_stack((audio_data, audio_data))

        stream.write(audio_data)
    # avoid hearing own output
#    time.sleep(0.5)
    wake_model.reset()
    sd.sleep(300)

  except Exception as e:
    print(f"[!] Speech Playback Error: {e}")

def listen_for_wake_word():
  """Dedicated Standby Function: Returns triggered model name when heard."""
  print("\n[💤 Standby Mode - Listening for Wake Word...]", flush=True)
  wake_model.reset()

  with sd.InputStream(
      device=config.DEVICE_INPUT,
      channels=1,
      samplerate=config.SAMPLE_RATE,
      blocksize=config.CHUNK_SIZE,
      dtype="int16",
  ) as stream:
    while True:
      data, _ = stream.read(config.CHUNK_SIZE)
      audio_chunk = data[:, 0]
      wake_model.predict(audio_chunk)

      for model_name, scores in wake_model.prediction_buffer.items():
        if scores[-1] >= config.WAKE_THRESHOLD:
          wake_model.reset()
          return model_name

def record_active_speech():
  """Dynamic VAD recording for active conversation mode."""
  print("\n[🎙️ Listening...] (Speak now)", flush=True)
  audio_buffer = []
  speech_detected = False
  silence_start = None
  start_time = time.time()

  def callback(indata, frames, time_info, status):
    nonlocal speech_detected, silence_start
    audio_buffer.append(indata.copy())
    volume = np.max(np.abs(indata))

    if volume > config.SILENCE_THRESHOLD:
      speech_detected = True
      silence_start = None
    elif speech_detected and silence_start is None:
      silence_start = time.time()

  with sd.InputStream(
      device=config.DEVICE_INPUT,
      samplerate=config.SAMPLE_RATE,
      channels=1,
      dtype="int16",
      callback=callback,
  ):
    while True:
      sd.sleep(100)
      now = time.time()

      if speech_detected and silence_start:
        if now - silence_start >= config.SILENCE_DURATION:
          print("[✓ Speech finished]")
          break

      if now - start_time >= config.MAX_RECORD_SECONDS:
        break

  if not speech_detected or not audio_buffer:
    return None

  audio_data = np.concatenate(audio_buffer, axis=0)
  wav_io = io.BytesIO()
  with wave.open(wav_io, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(config.SAMPLE_RATE)
    wf.writeframes(audio_data.tobytes())
  wav_io.seek(0)
  return wav_io
