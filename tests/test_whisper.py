import os
import sys
import requests

WHISPER_URL = "http://localhost:30900/v1/audio/transcriptions"
HEADERS = {"Authorization": "Bearer secret"}
TEST_FILE = "test.wav"

if not os.path.exists(TEST_FILE):
    print(f"[!] Error: {TEST_FILE} not found in the current directory.")
    sys.exit(1)

print(f"[*] Sending {TEST_FILE} to Whisper API at {WHISPER_URL}...")

try:
    with open(TEST_FILE, "rb") as f:
        files = {"file": (TEST_FILE, f, "audio/wav")}
        data = {
            "model": "Systran/faster-whisper-small.en",
            "language": "en"
        }
        
        response = requests.post(WHISPER_URL, files=files, data=data, timeout=15)
        
    print(f"[*] HTTP Status Code: {response.status_code}")
    print(f"[*] Raw Response Text: {response.text}")
    
    if response.status_code == 200:
        transcript = response.json().get("text", "").strip()
        print(f"\n[✔] SUCCESS! Transcribed Text: \"{transcript}\"")
    else:
        print(f"\n[!] Server returned error status {response.status_code}")

except requests.exceptions.ConnectionError:
    print(f"\n[!] Connection Error: Could not reach Whisper server at {WHISPER_URL}.")
    print("    Verify the pod status with: kubectl get pods")
except Exception as e:
    print(f"\n[!] Unexpected error: {e}")
