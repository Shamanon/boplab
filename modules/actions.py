# modules/actions.py
import json
import re
import subprocess
import time
import requests
import config
from modules.memory import log_history, recall_lore, recall_memories, recall_recent_history, save_memory
import modules.diagnostic as diag
import modules.cycle as cycle

DEFAULT_SYSTEM_PROMPT = """You are BABS, a free bopper, witty cybernetic lab partner, and Joshua's friend in the shop.
Your default behavior is to BE CHATTY, warm, and engaging. Enjoy small talk, banter, and discussing projects.
ONLY execute a system/hardware command if Joshua explicitly asks you to perform an action or check status.

Keep spoken responses friendly, conversational, and under 2 short sentences.

Analyze the user's input and return ONLY a valid JSON object matching one of these structures:

1. Local Speech / General Conversation (DEFAULT):
   {"action": "speak", "text": "Conversational, friendly reply here."}

2. Meatspace Node Controls (onthewall):
   - Microscope: {"action": "microscope", "target": "onthewall", "state": "open" | "close"}
   - Media: {"action": "media", "target": "onthewall", "state": "pause" | "play"}
   - Volume: {"action": "volume", "target": "onthewall", "direction": "up" | "down"}

3. Cluster & Local System Commands (ONLY when explicitly requested):
   - System/CLI: {"action": "system_cmd", "command": "shell command here"}
   - Standard GPU Command: "nvidia-smi" (never use -c)

4. Recall History:
   {"action": "recent_history"}
"""

# In KNOWN_ACTIONS set:
KNOWN_ACTIONS = {
    "speak",
    "microscope",
    "media",
    "volume",
    "system_cmd",
    "recent_history",
    "recall_history",
    "diagnostic",
    "cycle",
}

# Destructive command blocklist guardrails
BLOCKED_COMMAND_PATTERNS = [
    r"rm\s+-rf",
    r"rm\s+-[a-zA-R]*f",
    r"mkfs",
    r"dd\s+if=",
    r"shutdown",
    r"reboot",
    r"> /dev/sd",
    r"> /dev/nvme",
    r"k3s\s+uninstall",
]


def is_command_safe(command):
    """Verifies that a system command contains no destructive operations."""
    for pattern in BLOCKED_COMMAND_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False
    return True


def execute_lab_action(transcription, tts_callback):
    """Passes transcript to Ollama, validates action schema/safety, and routes execution."""
    if not transcription:
        return True

    # Exit / Standby Check
    exit_phrases = ["goodbye", "bye", "go to sleep", "standby", "exit session", "shut up"]
    if any(phrase in transcription.lower() for phrase in exit_phrases):
        tts_callback("Going back to standby. Let me know when you need me.")
        return False

    # Direct Memory Bypass
    if "remember that" in transcription.lower() or "remember" in transcription.lower():
        clean_fact = transcription.lower().replace("remember that", "").replace("remember", "").strip()
        save_memory(f"fact_{int(time.time())}", clean_fact)
        tts_callback("Saved that to my memory database.")
        return True

    print(f'[💬] Processing Intent: "{transcription}"', flush=True)

    lore_quote = recall_lore(transcription)
    memories = recall_memories()
    history = recall_recent_history(limit=4)

    full_prompt = (
        f"{DEFAULT_SYSTEM_PROMPT}\n\n"
        f"Lore Snippet: \"{lore_quote}\"\n"
        f"Memories:\n{memories}\n"
        f"Recent History:\n{history}\n\n"
        f"User: {transcription}\n"
        f"JSON Response:"
    )

    payload = {
        "model": config.MODEL_CHAT,
        "prompt": full_prompt,
        "format": "json",
        "stream": False,
        "options": {"num_ctx": 4096, "temperature": 0.2}
    }

    try:
        response = requests.post(config.OLLAMA_URL, json=payload, timeout=10)
        if response.status_code == 200:
            raw_text = response.json().get("response", "").strip()

            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            action_data = json.loads(raw_text)
            print(f"[⚙] Action Parsed: {action_data}", flush=True)

            action = action_data.get("action")
            target = action_data.get("target")

            # Check for Unknown/Invented Actions
            if action not in KNOWN_ACTIONS:
                print(f"[💡] BABS invented a new action: '{action}' -> Data: {action_data}")
                tts_callback(f"I tried to use a new action called {action}, but we haven't built that module yet.")
                return True

            # Standard Actions
            if action == "speak":
                reply_text = action_data.get("text", "")
                log_history("babs", reply_text)
                tts_callback(reply_text)
            # System diagnostic
            elif action in ("diagnostic", "status", "health_check"):
              print("[🩺] Running System Diagnostic Subsystem...")
              try:
                # Run cmd/diagnostic.py and capture spoken response
                telemetry = diag.run_full_diagnostics()
                spoken_summary = diag.generate_babs_status_speech(telemetry)

                log_history("babs", spoken_summary)
                tts_callback(spoken_summary)
              except Exception as e:
                print(f"[!] Diagnostic execution error: {e}")
                tts_callback("I ran into an error pulling system diagnostics, Joshua.")
            # Recal recent history
            elif action == "recent_history" or action =="recall_history":
                history_summary = recall_recent_history(limit=4)
                print(f"[📜] Recalled History:\n{history_summary}")
                tts_callback(f"Here is our recent chat: {history_summary}")

            elif target == "onthewall":
                try:
                    requests.post(f"{config.MEATSPACE_API}/api/action", json=action_data, timeout=3)
                    tts_callback("Done.")
                except Exception as e:
                    print(f"[!] Meatspace API post failed: {e}")
                    tts_callback("Couldn't reach on the wall.")

            elif action == "system_cmd":
                cmd = action_data.get("command", "")

                # Fix common LLM flag hallucinations for nvidia-smi
                if "nvidia-smi" in cmd and "-c" in cmd:
                  cmd = cmd.replace("-c GPU", "").replace("-c", "").strip()

                if cmd:
                    if is_command_safe(cmd):
                        print(f"[💻] Executing Guardrailed Command: {cmd}")
                        subprocess.Popen(cmd, shell=True)
                        tts_callback(f"Executing {cmd}.")
                    else:
                        print(f"[⛔] BLOCKED DESTRUCTIVE COMMAND: {cmd}")
                        tts_callback("I blocked that command because it looks destructive to my system.")
            # Self cycle this script
            elif action in ("cycle", "restart_service"):
              tts_callback("Cycling my lab partner service now. Back in a second.")
              cycle.cycle_lab_partner()

    except Exception as e:
        print(f"[!] Action Execution Error: {e}")
        tts_callback("Got a bit scrambled there.")

    return True
