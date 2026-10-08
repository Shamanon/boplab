#!/usr/bin/env python3
import json
import sys
import requests

OLLAMA_URL = "http://localhost:31434/api/generate"
# Using qwen2.5-coder:latest for fast, predictable responses
MODEL_NAME = "deepseek-r1:latest"


def test_ollama(prompt):
  print(f"Sending prompt to Ollama ({MODEL_NAME})...\n")

  payload = {
      "model": MODEL_NAME,
      "prompt": prompt,
      "system": "You are a reseaarch planning agent.",
      "stream": False,
      "options": {
          "temperature": 0.2,
      },
  }

  try:
    # 15-second strict timeout so it never hangs silently
    response = requests.post(OLLAMA_URL, json=payload, timeout=15)
    response.raise_for_status()

    result = response.json()
    output_text = result.get("response", "").strip()

    print("--- Ollama Response ---")
    print(output_text)
    print("-----------------------\n")
    return output_text

  except requests.exceptions.Timeout:
    print("[!] Error: Request timed out after 15 seconds.")
  except requests.exceptions.RequestException as e:
    print(f"[!] Error connecting to Ollama: {e}")
  return None


if __name__ == "__main__":
  user_prompt = (
      sys.argv[1]
      if len(sys.argv) > 1
      else "Explain the concept of GPU memory allocation in 2 short sentences."
  )
  test_ollama(user_prompt)
