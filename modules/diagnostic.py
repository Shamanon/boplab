#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import requests


def get_uptime():
  try:
    with open("/proc/uptime", "r") as f:
      uptime_seconds = float(f.readline().split()[0])
      hours = int(uptime_seconds // 3600)
      minutes = int((uptime_seconds % 3600) // 60)
      return f"{hours}h {minutes}m"
  except Exception:
    return "Unknown"


def get_gpu_status():
  try:
    cmd = "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"
    output = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
    gpus = []
    for line in output.split("\n"):
      if line:
        parts = [p.strip() for p in line.split(",")]
        gpus.append({
            "name": parts[0],
            "util_pct": parts[1],
            "vram_used_mb": parts[2],
            "vram_total_mb": parts[3],
            "temp_c": parts[4],
        })
    return gpus
  except Exception:
    return []


def get_disk_usage(path="/var"):
  try:
    total, used, free = shutil.disk_usage(path)
    return {
        "total_gb": round(total / (1024**3), 1),
        "free_gb": round(free / (1024**3), 1),
        "used_pct": round((used / total) * 100, 1),
    }
  except Exception:
    return {}


def run_full_diagnostics():
  uptime = get_uptime()
  gpus = get_gpu_status()
  disk = get_disk_usage("/var")

  # Format summary telemetry
  summary_lines = [f"Uptime: {uptime}"]
  if disk:
    summary_lines.append(
        f"/var Drive: {disk['free_gb']}GB free out of {disk['total_gb']}GB"
        f" ({disk['used_pct']}% used)"
    )

  if gpus:
    for idx, g in enumerate(gpus):
      summary_lines.append(
          f"GPU {idx} ({g['name']}): {g['util_pct']}% Load, {g['temp_c']}°C,"
          f" VRAM {g['vram_used_mb']}/{g['vram_total_mb']}MB"
      )

  raw_telemetry = "\n".join(summary_lines)
  return raw_telemetry


def generate_babs_status_speech(telemetry_text):
  """Pipes telemetry through Ollama so BABS speaks a natural, short diagnostic summary."""
  prompt = f"""You are BABS reporting system status to Joshua in the lab.
Below is real system telemetry data:
{telemetry_text}

Summarize this status naturally in under 2 short conversational sentences. Mention overall health and any notable metrics like GPU load or free disk space."""

  try:
    res = requests.post(
        "http://localhost:31434/api/generate",
        json={
            "model": "babs-cyberpunk",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3},
        },
        timeout=30,
    )
    if res.status_code == 200:
      return res.json().get("response", "").strip()
  except Exception as e:
    print(f"[!] Ollama status synthesis failed: {e}")

  # Fallback spoken output if Ollama is busy
  return f"Systems operational. Uptime is {get_uptime()}. Hardware telemetry collected cleanly."


if __name__ == "__main__":
  telemetry = run_full_diagnostics()
  print("--- RAW TELEMETRY ---")
  print(telemetry)
  print("\n--- BABS SPOKEN REPORT ---")
  print(generate_babs_status_speech(telemetry))
