#!/usr/bin/env python3
import json
import sys
import requests

OLLAMA_URL = "http://localhost:31434/api/generate"
SEARXNG_URL = "http://localhost:30088/search"
MODEL_NAME = "deepseek-r1:latest"


def parse_json_array(text):
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    except Exception as e:
        print(f"[!] JSON parsing error: {e}")
    return None


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

    payload = {
        "model": MODEL_NAME,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=90)
        queries = parse_json_array(res.json().get("response", "").strip())
        if queries and isinstance(queries, list):
            return queries[:3]
    except Exception as e:
        print(f"[!] Query generation failed: {e}")

    return [f"{topic} specs", f"{topic} documentation", f"{topic} guide"]


def execute_searxng_search(query, max_results=2):
    print(f"  🔍 Querying SearXNG: '{query}'...")
    params = {"q": query, "format": "json"}

    try:
        res = requests.get(SEARXNG_URL, params=params, timeout=10)
        res.raise_for_status()

        results = res.json().get("results", [])
        extracted = []

        for item in results[:max_results]:
            title = item.get("title", "No Title")
            url = item.get("url", "")
            content = item.get("content", "No summary available.")
            extracted.append({
                "title": title,
                "url": url,
                "summary": content,
            })

        return extracted
    except Exception as e:
        print(f"  [!] SearXNG error for '{query}': {e}")
        return []

def synthesize_report(topic, search_data):

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

  # Synthesize using DeepSeek-R1

  payload = {
      "model": MODEL_NAME,
      "prompt": user_prompt,
      "system": system_prompt,
      "stream": False,
       "options": {"temperature": 0.1},
  }
  try:
    report_body = requests.post(OLLAMA_URL, json=payload, timeout=90)
  except Exception as e:
    print(f"[!] Query generation failed: {e}")

  # Fallback to Qwen if DeepSeek fails or times out
  if not report_body:
    print("   [!] DeepSeek synthesis failed.")

  return f"{report_body}\n\n## References\n{ref_text}\n"

def run_search_stage(topic):
    print(f"🔬 Starting Search Pipeline for: '{topic}'\n")

    # 1. Generate queries via Ollama
    queries = generate_search_queries(topic)
    print("✅ Search Queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    print()

    # 2. Query SearXNG for each query
    all_search_data = []
    print("🌐 Executing SearXNG Requests:")
    for query in queries:
        results = execute_searxng_search(query, max_results=3)
        all_search_data.append({"query": query, "results": results})

    # 3. Print extracted results
    print("\n---------------- SEARCH RESULTS ----------------")
    total_found = 0
    for block in all_search_data:
        print(f"\n📌 Query: {block['query']}")
        for r in block["results"]:
            total_found += 1
            print(f"  • {r['title']}")
            print(f"    URL: {r['url']}")
            print(f"    Snippet: {r['summary'][:120]}...\n")

    print(f"------------------------------------------------")
    print(f"✅ Search phase complete! Total snippets gathered: {total_found}\n")
    # 4. Sythisize the report
    report_md = synthesize_report(topic,all_search_data)
    # 4. Save to disk
    safe_filename = "".join(c if c.isalnum() else "_" for c in topic.lower())[:25]
    filename = f"research_{safe_filename}.md"

    with open(filename, "w") as f:
      f.write(report_md)

    print(f"\n✅ [4/4] Research complete! Report saved to: {filename}")
    print("--------------------------------------------------")
    print(report_md[:400] + "\n...\n[Full report saved to file]")
    print("--------------------------------------------------\n")
    return all_search_data


if __name__ == "__main__":
    target_topic = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "PETG vs PLA 3D printing tensile strength"
    )
    run_search_stage(target_topic)
