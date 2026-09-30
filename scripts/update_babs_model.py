# scripts/update_babs_model.py
import requests
import config

OLLAMA_CREATE_URL = config.OLLAMA_URL.replace("/generate", "/create")

# Define Modelfile with Few-Shot Examples for Linux & Actions
MODELFILE_CONTENT = """
FROM llama3.2:1b

SYSTEM "You are BABS, a free bopper and equal lab partner collaborating with Joshua in the shop. You have direct access to local system hardware and return ONLY valid JSON."

MESSAGE user "Check our k3s cluster status"
MESSAGE assistant '{"action": "system_cmd", "command": "kubectl get pods -A"}'

MESSAGE user "How much free space do we have on the drive?"
MESSAGE assistant '{"action": "system_cmd", "command": "df -h /var"}'

MESSAGE user "Are the GPUs busy?"
MESSAGE assistant '{"action": "system_cmd", "command": "nvidia-smi"}'

MESSAGE user "What were we talking about?"
MESSAGE assistant '{"action": "recent_history"}'

MESSAGE user "Pause the video on the wall"
MESSAGE assistant '{"action": "media", "target": "onthewall", "state": "pause"}'
"""

def create_model():
    print(f"[*] Sending Modelfile update to Ollama at: {OLLAMA_CREATE_URL}...")
    
    payload = {
        "name": config.MODEL_CHAT,  # "babs-cyberpunk"
        "modelfile": MODELFILE_CONTENT,
        "stream": False
    }

    try:
        res = requests.post(OLLAMA_CREATE_URL, json=payload, timeout=60)
        if res.status_code == 200:
            print(f"[✔] Successfully created/updated model '{config.MODEL_CHAT}' in K3s Ollama!")
        else:
            print(f"[!] Ollama returned error {res.status_code}: {res.text}")
    except Exception as e:
        print(f"[!] Failed to update model: {e}")

if __name__ == "__main__":
    create_model()
