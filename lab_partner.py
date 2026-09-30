# lab_partner.py
#import io
#import os
#import shlex
#import subprocess
#import time
#import wave
#import numpy as np
#import onnxruntime as ort
#from piper import PiperVoice
#from openwakeword.model import Model
import config

from modules.actions import execute_lab_action
from modules.memory import log_history
from modules.stt import transcribe_audio

print("[*] Booting B0P-L@B Core Engine...")

from modules.audio import babs_speak, listen_for_wake_word, record_active_speech
#import sounddevice as sd

def main():
  print("\n==================================================")
  print(" boplab // lab_partner.py online")
  print("==================================================\n")

  while True:
    # 1. Standby Mode
    trigger = listen_for_wake_word()

    if "bebopalula" in trigger:
      print("\n[!] BEBOPALULA TRIGGERED!")
      babs_speak("Be-bop-a-lula she's my baby! Sequence initiated!")
      continue

    print(f"\n[🔔 Wake Word Detected: {trigger}]")
    babs_speak("Hey! What's up?")

    # 2. Active Session Loop
    active_session = True
    consecutive_silence = 0

    while active_session:
      audio_wav = record_active_speech()

      if not audio_wav:
        consecutive_silence += 1
        if consecutive_silence >= 2:  # Exit session after ~2 silent attempts
          babs_speak("Going back to standby.")
          active_session = False
        continue

      consecutive_silence = 0
      text = transcribe_audio(audio_wav)

      if text:
        log_history("user", text)
        active_session = execute_lab_action(text, tts_callback=babs_speak)

#      wake_model.reset()
#      sd.sleep(300)


if __name__ == "__main__":
  main()
