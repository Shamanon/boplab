#!/usr/bin/env python3
import json
import sys
import requests

OLLAMA_URL = "http://localhost:31434/api/generate"
MODEL_NAME = "qwen2.5-coder:latest"


def parse_json_array(text):
  """Extracts a valid JSON array from raw LLM output, handling markdown blocks."""
  try:
    # Locate the first '[' and last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
      array_str = text[start : end + 1]
      return json.loads(array_str)
  except Exception as e:
    print(f"[!] JSON parsing error: {e}")
  return None


def generate_search_queries(topic):
  print(f"Generating 3 search queries for topic: '{topic}'...\n")

  system_prompt = (
      "You are an expert research assistant. You MUST respond ONLY with a raw"
      " JSON list of 3 strings."
  )
  user_prompt = (
      f'Create 3 distinct search engine queries to gather deep technical'
      f' details about: "{topic}".\n\n'
      "EXAMPLE OUTPUT FORMAT:\n"
      '["query 1", "query 2", "query 3"]\n\n'
      "DO NOT include markdown, commentary, or extra text."
  )

  payload = {
      "model": MODEL_NAME,
      "prompt": user_prompt,
      "system": system_prompt,
      "stream": False,
      "options": {"temperature": 0.1},  # Low temp for precise formatting
  }

  try:
    res = requests.post(OLLAMA_URL, json=payload, timeout=20)
    res.raise_for_status()

    raw_output = res.json().get("response", "").strip()
    print(f"--- Raw Model Output ---\n{raw_output}\n------------------------\n")

    queries = parse_json_array(raw_output)

    if queries and isinstance(queries, list):
      print("✅ Successfully parsed search queries:")
      for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
      return queries
    else:
      print(
          "[!] Warning: Could not parse JSON array. Falling back to default"
          " query list."
      )
      fallback = [
          f"{topic} technical specs",
          f"{topic} documentation",
          f"{topic} guide",
      ]
      return fallback

  except Exception as e:
    print(f"[!] Request failed: {e}")
    return None


if __name__ == "__main__":
  target_topic = (
      sys.argv[1]
      if len(sys.argv) > 1
      else "PETG vs PLA 3D printing tensile strength"
  )
  generate_search_queries(target_topic)
