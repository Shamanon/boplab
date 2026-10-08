#!/usr/bin/env python3
import json
import os
import re
import sys
from urllib.parse import urlparse
import requests

# Optional dependency for reading PDFs
try:
  import pypdf

  HAS_PYPDF = True
except ImportError:
  HAS_PYPDF = False

# Endpoints & Model Settings
OLLAMA_URL = "http://localhost:31434/api/generate"
SEARXNG_URL = "http://localhost:30088/search"
MODEL_NAME = "deepseek-r1:latest"

# Configuration Constraints
NUM_QUERIES = 5  # Expanded query count
RESULTS_PER_QUERY = 10  # Web snippets per query
MAX_PDFS_TO_FETCH = 10  # Max PDFs to download/parse
MAX_IMAGES_TO_SAVE = 10  # Max images to retrieve and download


def strip_think_tags(text):
  """Strips DeepSeek R1 reasoning blocks using ASCII escapes for XML tags."""
  tag_start = "\x3cthink\x3e"  # 
  tag_end = "\x3c/think\x3e"  # 

  while tag_start in text and tag_end in text:
    start = text.find(tag_start)
    end = text.find(tag_end) + len(tag_end)
    text = text[:start] + text[end:]
  return text.strip()


def parse_json_array(text):
  try:
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
      return json.loads(text[start : end + 1])
  except Exception as e:
    print(f"[!] JSON parse error: {e}")
  return None


def query_ollama(prompt, system_prompt="", timeout=180):
  payload = {
      "model": MODEL_NAME,
      "prompt": prompt,
      "system": system_prompt,
      "stream": False,
      "options": {"temperature": 0.2},
      "num_predict": 4096,
  }
  headers = {"Connection": "close"}

  try:
    with requests.Session() as session:
      res = session.post(
          OLLAMA_URL, json=payload, headers=headers, timeout=timeout
      )
      res.raise_for_status()
      raw_bytes = res.content.decode("utf-8", errors="ignore")
      response_data = json.loads(raw_bytes)
      return strip_think_tags(response_data.get("response", "").strip())
  except Exception as e:
    print(f"[!] Ollama call failed: {e}")
    return ""


def generate_search_queries(topic):
  system_prompt = (
      "You are an expert research assistant. You MUST respond ONLY with a raw"
      f" JSON list of {NUM_QUERIES} strings."
  )
  user_prompt = (
      f'Create {NUM_QUERIES} distinct, highly specific search engine queries'
      f' to gather technical specs, diagrams, and documentation about:'
      f' "{topic}".\n\n'
      'EXAMPLE OUTPUT FORMAT:\n["query 1", "query 2", "query 3", "query 4",'
      ' "query 5"]\n\n'
      "DO NOT include markdown, explanations, or extra text."
  )

  raw_out = query_ollama(user_prompt, system_prompt, timeout=90)
  queries = parse_json_array(raw_out)

  if queries and isinstance(queries, list):
    return queries[:NUM_QUERIES]

  print("   [!] Query parse failed. Using fallback queries.")
  return [
      f"{topic} technical specs",
      f"{topic} documentation",
      f"{topic} architecture diagram",
      f"{topic} filetype:pdf",
      f"{topic} whitepaper",
  ]


def execute_searxng_search(query, categories="general", max_results=5):
  print(f"  🔍 Querying SearXNG ({categories}): '{query}'...")
  params = {"q": query, "format": "json", "categories": categories}

  try:
    res = requests.get(SEARXNG_URL, params=params, timeout=12)
    res.raise_for_status()
    results = res.json().get("results", [])
    return results[:max_results]
  except Exception as e:
    print(f"  [!] SearXNG error for '{query}': {e}")
    return []


def process_pdf(pdf_url, pdf_dir):
  """Downloads a PDF, extracts up to 3 pages of text, and saves the file."""
  if not HAS_PYPDF:
    return ""

  filename = os.path.basename(urlparse(pdf_url).path)
  if not filename.endswith(".pdf"):
    filename = "document.pdf"

  safe_filename = "".join(c if c.isalnum() else "_" for c in filename[:-4]) + ".pdf"
  file_path = os.path.join(pdf_dir, safe_filename)

  try:
    print(f"  📄 Fetching PDF: {pdf_url}")
    res = requests.get(
        pdf_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}
    )
    res.raise_for_status()

    with open(file_path, "wb") as f:
      f.write(res.content)

    reader = pypdf.PdfReader(file_path)
    extracted_text = ""
    # Extract text from the first 3 pages
    for page in reader.pages[:3]:
      extracted_text += page.extract_text() + "\n"

    print(f"     ✅ Saved PDF to {safe_filename} ({len(extracted_text)} chars extracted)")
    return f"\n--- PDF CONTENT ({safe_filename}) ---\n{extracted_text[:2000]}\n"
  except Exception as e:
    print(f"     [!] Failed to download/parse PDF: {e}")
    return ""


def process_images(topic, img_dir):
  """Searches SearXNG for images related to the topic and downloads thumbnails/images."""
  print(f"\n🖼️  Gathering images for: '{topic}'...")
  results = execute_searxng_search(
      f"{topic} diagram", categories="images", max_results=MAX_IMAGES_TO_SAVE
  )
  downloaded_images = []

  for i, img in enumerate(results, 1):
    img_url = img.get("img_src") or img.get("thumbnail_src") or img.get("url")
    if not img_url:
      continue

    ext = os.path.splitext(urlparse(img_url).path)[1]
    if ext.lower() not in [".jpg", ".jpeg", ".png", ".webp", ".svg"]:
      ext = ".jpg"

    filename = f"image_{i}{ext}"
    file_path = os.path.join(img_dir, filename)

    try:
      res = requests.get(
          img_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}
      )
      res.raise_for_status()
      with open(file_path, "wb") as f:
        f.write(res.content)

      title = img.get("title", f"Image {i}")
      downloaded_images.append(
          {"filename": filename, "rel_path": f"images/{filename}", "title": title}
      )
      print(f"  ✅ Saved image: {filename}")
    except Exception as e:
      print(f"  [!] Failed to download image {i}: {e}")

  return downloaded_images


def run_robust_deep_research(topic):
  # 1. Setup Output Directory Structure
  safe_dir_name = "".join(c if c.isalnum() else "_" for c in topic.lower())[:30]
  output_dir = os.path.join("research_output", safe_dir_name)
  pdf_dir = os.path.join(output_dir, "pdfs")
  img_dir = os.path.join(output_dir, "images")

  os.makedirs(pdf_dir, exist_ok=True)
  os.makedirs(img_dir, exist_ok=True)

  print(f"🔬 Starting Robust Deep Research Pipeline for: '{topic}'")
  print(f"📁 Output Directory: {output_dir}\n")

  # 2. Query Generation
  print("⚙️  [1/4] Generating expanded search strategy...")
  queries = generate_search_queries(topic)
  print("   Generated Queries:")
  for i, q in enumerate(queries, 1):
    print(f"     {i}. {q}")
  print()

  # 3. Web Data & PDF Harvesting
  print("🌐 [2/4] Executing web & document searches via SearXNG...")
  gathered_notes = []
  references = []
  pdf_count = 0

  for q in queries:
    results = execute_searxng_search(
        q, categories="general", max_results=RESULTS_PER_QUERY
    )

    for item in results:
      url = item.get("url", "")
      title = item.get("title", "Untitled")
      snippet = item.get("content", "")

      gathered_notes.append(
          f"### {title}\n**URL:** {url}\n**Snippet:** {snippet}\n"
      )
      references.append(f"- [{title}]({url})")

      # Download PDFs if found and under quota
      if (
          url.lower().endswith(".pdf")
          and pdf_count < MAX_PDFS_TO_FETCH
          and HAS_PYPDF
      ):
        pdf_text = process_pdf(url, pdf_dir)
        if pdf_text:
          gathered_notes.append(pdf_text)
          pdf_count += 1

  # 4. Image Harvesting
  saved_images = process_images(topic, img_dir)

  # 5. Synthesize Markdown Report
  print("\n📝 [3/4] Synthesizing comprehensive report with DeepSeek-R1...")
  notes_block = "\n".join(gathered_notes)
  refs_block = "\n".join(list(set(references)))

  system_prompt = "You are an expert research scientist and technical writer."
  user_prompt = f"""Synthesize an exhaustive, publication-ready technical research report of at least 1,500 words.
Topic: "{topic}"

Gathered Web & Document Context:
{notes_block}

Format the output in clean Markdown using these sections:
# {topic}

## Executive Summary
## Architecture Overview
## Detailed Technical Analysis
## Comparative Trade-offs & Specifications
## Implementation & Architecture Recommendations
## References

Ensure technical accuracy and cite the provided web sources where applicable."""

  report_body = query_ollama(user_prompt, system_prompt, timeout=240)

  # Attach local image embeds to the Markdown report if present
  image_md_block = ""
  if saved_images:
    image_md_block = "\n\n## Visual References & Diagrams\n"
    for img in saved_images:
      image_md_block += f"![{img['title']}]({img['rel_path']})\n*{img['title']}*\n\n"

  final_report = (
      f"{report_body}\n{image_md_block}\n## References\n{refs_block}\n"
  )

  # 6. Write Final Output
  report_filepath = os.path.join(output_dir, "report.md")
  with open(report_filepath, "w") as f:
    f.write(final_report)

  print(f"\n✅ [4/4] Research complete!")
  print(f"📄 Report saved to: {report_filepath}")
  print(f"📁 PDF assets saved in: {pdf_dir}/")
  print(f"🖼️  Image assets saved in: {img_dir}/\n")


if __name__ == "__main__":
  target_topic = (
      sys.argv[1]
      if len(sys.argv) > 1
      else "PETG vs PLA 3D printing tensile strength"
  )

  # Install pypdf automatically if missing
  if not HAS_PYPDF:
    print("[!] 'pypdf' module not found. Installing now...")
    os.system(f"{sys.executable} -m pip install pypdf -q")
    try:
      import pypdf

      HAS_PYPDF = True
    except ImportError:
      print("[!] Could not auto-install pypdf. PDF parsing will be skipped.")

  run_robust_deep_research(target_topic)
