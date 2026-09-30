from flask import Flask, request, jsonify
import subprocess
import os
import signal

app = Flask(__name__)

# Track ffplay process ID so we can cleanly launch/kill it
ffplay_process = None
media_was_playing = False

MICROSCOPE_DEV = "/dev/video0"  # Adjust if microscope enumerates as /dev/video1 or /dev/video2

def speak(text):
    """Speaks text using espeak-ng through onthewall's speakers."""
    subprocess.Popen(["espeak-ng", "-v", "en-us", "-s", "160", text])

def pause_media():
    """Pauses media players (Spotify, VLC, Web browsers) via playerctl."""
    global media_was_playing
    try:
        status = subprocess.check_output(["playerctl", "status"], text=True).strip()
        if status == "Playing":
            media_was_playing = True
            subprocess.run(["playerctl", "pause"])
    except Exception:
        media_was_playing = False

def resume_media():
    """Resumes media if it was paused during the interaction."""
    global media_was_playing
    if media_was_playing:
        subprocess.run(["playerctl", "play"])
        media_was_playing = False

@app.route('/api/pause', methods=['POST'])
def handle_pause():
    """Fired immediately when BABS hears 'Hey BABS' before processing speech."""
    pause_media()
    speak("What's up?")
    return jsonify({"status": "audio_paused", "acknowledged": True})

@app.route('/api/action', methods=['POST'])
def handle_action():
    """Receives parsed JSON action payload from BABS Ollama engine."""
    global ffplay_process
    data = request.json or {}
    action = data.get("action")
    state = data.get("state", "open")
    text_to_say = data.get("text")

    print(f"[*] Received Action from BABS: {data}")

    # Say / Verbal Feedback Action
    if action == "say" or text_to_say:
        speak(text_to_say or "Command received.")

    # Microscope Action via ffplay
    elif action == "microscope":
        if state in ["open", "start", "show"]:
            # Check if ffplay is already running
            if ffplay_process is None or ffplay_process.poll() is not None:
                speak("Opening microscope feed.")
                env = os.environ.copy()
                env["DISPLAY"] = ":0"  # Directs ffplay to onthewall's main desktop display
                ffplay_process = subprocess.Popen(
                    [
                        "ffplay",
                        "-f",
                        "v4l2",
                        "-video_size",
                        "1280x720",
                        "-framerate",
                        "30",
                        "-window_title",
                        "Lab Microscope Feed",
                        "-noborder",
                        MICROSCOPE_DEV,
                    ],
                    env=env,
                )
                # Launch ffplay in low-latency mode
#                ffplay_process = subprocess.Popen([
#                    "ffplay",
#                    "-f", "v4l2",
#                    "-video_size", "1280x720",
#                    "-framerate", "30",
#                    "-window_title", "Lab Microscope Feed",
#                    "-noborder",
#                    MICROSCOPE_DEV
#                ])
            else:
                speak("Microscope is already open.")

        elif state in ["close", "stop", "hide"]:
            if ffplay_process and ffplay_process.poll() is None:
                speak("Closing microscope feed.")
                ffplay_process.terminate()
                ffplay_process = None
                resume_media()
            else:
                # Fallback kill via pkill if process handle was lost
                subprocess.run(["pkill", "-f", "ffplay.*v4l2"])
                speak("Closed microscope feed.")
                resume_media()

    # Resume Media Action
    elif action == "resume":
        speak("Resuming playback.")
        resume_media()

    # Volume Control Action
    elif action == "volume":
      direction = data.get("direction", "up")
      if direction == "up":
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"])
      elif direction == "down":
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"])

    # General Media Action
    elif action == "media":
      state = data.get("state")
      if state == "pause":
        subprocess.run(["playerctl", "pause"])
      elif state == "play" or state == "resume":
        subprocess.run(["playerctl", "play"])

    return jsonify({"status": "success", "action_processed": action})

if __name__ == '__main__':
    # Listen on all interfaces so BABS can hit port 5000 over local LAN
    app.run(host='0.0.0.0', port=5000)
