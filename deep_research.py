#!/usr/bin/env python3
import json
import requests
import sys

OLLAMA_URL = "http://localhost:31434/api/generate"
SEARXNG_URL = "http://localhost:30088/search"
MODEL_NAME = "deepseek-r1:latest"


def strip_think_tags(text):
  """Strips out DeepSeek R1 reasoning blocks cleanly using basic string slicing."""
  while "<think>" in text and "</think>" in text:
    start = text.find("")
    end = text.find("") + len("")
    text = text[:start] + text[end:]
  return text.strip()


def parse_json_array(text):
  try:
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
      return json.loads(text[start : end + 1])
  except Exception as e:
    print(f"[!] JSON parsing error: {e}")
  return None


def query_ollama(prompt, system_prompt="", timeout=120):
  payload = {
      "model": MODEL_NAME,
      "prompt": prompt,
      "system": system_prompt,
      "stream": False,
      "options": {"temperature": 0.2},
  }
  headers = {"Connection": "close"}

  try:
    with requests.Session() as session:
      res = session.post(
          OLLAMA_URL, json=payload, headers=headers, timeout=timeout
      )
      res.raise_for_status()

      raw_bytes = res.content
      decoded_text = raw_bytes.decode("utf-8", errors="ignore")
      response_data = json.loads(decoded_text)

      raw_text = response_data.get("response", "").strip()
      return strip_think_tags(raw_text)
  except Exception as e:
    print(f"[!] Ollama query failed for model {MODEL_NAME}: {e}")
    return ""


def generate_search_queries(topic):
  system_prompt = (
      "You are an expert research assistant. You MUST respond ONLY with a raw"
      " JSON list of 3 strings."
  )
  user_prompt = (
      f'Create 3 distinct search engine queries to gather deep technical'
      f' details about: "{topic}".\n\n'
      'EXAMPLE OUTPUT FORMAT:\n["query 1", "query 2", "query 3"]\n\n'
      "DO NOT include markdown or extra text."
  )

  raw_out = query_ollama(user_prompt, system_prompt, timeout=90)
  queries = parse_json_array(raw_out)

  if queries and isinstance(queries, list):
    return queries[:3]

  print(
      "   [!] Could not parse LLM queries. Falling back to default query list."
  )
  return [f"{topic} specs", f"{topic} documentation", f"{topic} guide"]


def execute_searxng_search(query, max_results=3):
  print(f"  🔍 Querying SearXNG: '{query}'...")
  params = {"q": query, "format": "json"}

  try:
    res = requests.get(SEARXNG_URL, params=params, timeout=10)
    res.raise_for_status()

    results = res.json().get("results", [])
    extracted = []

    for item in results[:max_results]:
      extracted.append({
          "title": item.get("title", "No Title"),
          "url": item.get("url", ""),
          "summary": item.get("content", "No summary available."),
      })
    return extracted
  except Exception as e:
    print(f"  [!] SearXNG error for '{query}': {e}")
    return []


def synthesize_report(topic, search_data):
  print(
      "\n📝 [3/4] Synthesizing gathered technical notes with DeepSeek-R1..."
  )

  formatted_notes = []
  references = []

  for block in search_data:
    for item in block["results"]:
      formatted_notes.append(
          f"### {item['title']}\n**URL:** {item['url']}\n**Snippet:**"
          f" {item['summary']}\n"
      )
      references.append(f"- [{item['title']}]({item['url']})")

  notes_text = "\n".join(formatted_notes)
  ref_text = "\n".join(list(set(references)))

  system_prompt = "You are a master technical writer and research engineer."
  user_prompt = f"""Synthesize a deep, comprehensive technical report on the following topic:
Topic: "{topic}"

Gathered Web Research Notes:
{notes_text}

Format the report using clean Markdown with the following structure:
# {topic}

## Executive Summary
## Technical Analysis & Key Findings
## Implementation / Practical Recommendations
## References

Ensure all insights are backed by the provided web sources."""

  report_body = query_ollama(user_prompt, system_prompt, timeout=180)
  return f"{report_body}\n\n## References\n{ref_text}\n"


def run_deep_research(topic):
  print(f"🔬 Starting Deep Research Pipeline for: '{topic}'\n")

  # 1. Generate Queries
  print("⚙️  [1/4] Generating search strategy...")
  queries = generate_search_queries(topic)
  print("   Queries generated:")
  for i, q in enumerate(queries, 1):
    print(f"     {i}. {q}")
  print()

  # 2. Gather Web Data
  print("🌐 [2/4] Executing web searches via local SearXNG...")
  search_data = []
  for q in queries:
    results = execute_searxng_search(q, max_results=3)
    search_data.append({"query": q, "results": results})

  # 3. Synthesize Report
  report_md = synthesize_report(topic, search_data)

  # 4. Save to disk
  safe_filename = "".join(c if c.isalnum() else "_" for c in topic.lower())[:25]
  filename = f"research_{safe_filename}.md"

  with open(filename, "w") as f:
    f.write(report_md)

  print(f"\n✅ [4/4] Research complete! Report saved to: {filename}")
  print("--------------------------------------------------")
  print(report_md[:400] + "\n...\n[Full report saved to file]")
  print("--------------------------------------------------\n")


if __name__ == "__main__":
  target_topic = (
      sys.argv[1]
      if len(sys.argv) > 1
      else "PETG vs PLA 3D printing tensile strength"
  )
  run_deep_research(target_topic)
