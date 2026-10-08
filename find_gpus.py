#!/usr/bin/env python3
import json
import re
import requests
from typing import List, Dict, Any

# ==========================================
# CONFIGURATION
# ==========================================
SEARXNG_URL = "http://localhost:30088"  # Update to your SearXNG instance endpoint
OLLAMA_URL = "http://localhost:31434"   # Update to your Ollama endpoint
OLLAMA_MODEL = "deepseek-r1:latest"         # Or llama3.1, mistral, etc.

# Whitelist of mainstream GPUs meeting CUDA Compute Capability >= 7.5
VALID_CUDA_75_MODELS = [
    # RTX 20 Series (Turing - CC 7.5)
    "2060", "2070", "2080", "2080 ti", "titan rtx",
    # GTX 16 Series (Turing - CC 7.5)
    "1650", "1660", "1660 super", "1660 ti",
    # Workstation / Data Center Turing (CC 7.5)
    "quadro rtx 3000", "quadro rtx 4000", "quadro rtx 5000", "quadro rtx 6000", "quadro rtx 8000",
    "t400", "t600", "t1000", "t1200", "tesla t4",
    # RTX 30 Series (Ampere - CC 8.6)
    "3050", "3060", "3060 ti", "3070", "3070 ti", "3080", "3080 ti", "3090", "3090 ti",
    # RTX 40 Series (Ada Lovelace - CC 8.9)
    "4050", "4060", "4060 ti", "4070", "4070 super", "4070 ti", "4080", "4090",
    # RTX 50 Series (Blackwell - CC 10.x/12.x)
    "5060", "5070", "5080", "5090"
]

# Blacklist of common Pascal/Maxwell/older models (CC < 7.5) to reject explicitly
INVALID_MODELS = [
    "1050", "1060", "1070", "1080", "960", "970", "980", "750", "780", "titan x", "titan xp", "p4000", "p5000"
]

SEARCH_QUERIES = [
    "used nvidia rtx graphic card buy ebay marketplace",
    "used nvidia gtx 1660 2060 3060 graphics card for sale",
    "cheap used nvidia rtx 2070 2080 3070 GPU ebay craigslist"
]

# ==========================================
# SEARCH & OLLAMA HELPERS
# ==========================================
def search_searxng(query: str) -> List[Dict[str, Any]]:
    """Query local SearXNG engine for web listings."""
    params = {
        "q": query,
        "format": "json",
        "pageno": 1
    }
    try:
        resp = requests.get(f"{SEARXNG_URL}/search", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as e:
        print(f"[!] SearXNG Search Error: {e}")
        return []

def strip_think_tags(text: str) -> str:
    """Strip reasoning blocks from models like DeepSeek-R1."""
    return re.sub(r".*?", "", text, flags=re.DOTALL).strip()

def extract_price_and_model_via_ollama(title: str, snippet: str) -> Dict[str, Any]:
    """Uses Ollama to structure raw web snippet into clean GPU model and price."""
    prompt = f"""
    Analyze this web listing snippet for a used NVIDIA graphics card:
    Title: "{title}"
    Snippet: "{snippet}"

    Return JSON ONLY with two fields:
    - "gpu_model": (string, e.g., "RTX 2060", "GTX 1080 Ti", "RTX 3070", or "Unknown")
    - "price_usd": (float or null if price not found)

    JSON:
    """
    
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=15
        )
        resp.raise_for_status()
        raw_response = resp.json().get("response", "")
        clean_json = strip_think_tags(raw_response)
        return json.loads(clean_json)
    except Exception:
        # Fallback regex parser if LLM fails or times out
        price_match = re.search(r'\$(\d+(?:\.\d{2})?)', f"{title} {snippet}")
        price = float(price_match.group(1)) if price_match else None
        return {"gpu_model": title, "price_usd": price}

def is_cuda_75_plus(gpu_model_str: str) -> bool:
    """Validates whether the detected GPU model meets CUDA CC >= 7.5."""
    model_lower = gpu_model_str.lower()
    
    # Reject explicitly if it matches an older card generation
    if any(inv in model_lower for inv in INVALID_MODELS) and not any(valid in model_lower for valid in ["2080", "3080", "4080"]):
        return False
        
    # Check if any valid CC >= 7.5 keyword exists in the string
    return any(valid in model_lower for valid in VALID_CUDA_75_MODELS)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Starting Used GPU Search Pipeline...")
    raw_results = []
    
    for q in SEARCH_QUERIES:
        print(f"[*] Querying SearXNG: {q}")
        results = search_searxng(q)
        raw_results.extend(results)

    valid_listings = []
    seen_urls = set()

    print(f"[*] Processing {len(raw_results)} total retrieved search results...")
    
    for item in raw_results:
        url = item.get("url")
        title = item.get("title", "")
        snippet = item.get("content", "")

        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        # Quick pre-filter: Ignore non-NVIDIA listings early
        if not re.search(r'nvidia|rtx|gtx|quadro|tesla', f"{title} {snippet}", re.IGNORECASE):
            continue

        # LLM extraction step
        parsed = extract_price_and_model_via_ollama(title, snippet)
        gpu_model = parsed.get("gpu_model", "Unknown")
        price = parsed.get("price_usd")

        # Skip if model doesn't satisfy CUDA >= 7.5 or price wasn't identifiable
        if not price or not is_cuda_75_plus(gpu_model):
            continue

        valid_listings.append({
            "model": gpu_model,
            "price": float(price),
            "title": title,
            "url": url,
            "engine": item.get("engine", "web")
        })

    # Sort listings by cheapest price
    sorted_listings = sorted(valid_listings, key=lambda x: x["price"])

    print("\n" + "="*80)
    print(f" FOUND {len(sorted_listings)} VALID GPU LISTINGS (CUDA COMPUTE CAPABILITY >= 7.5)")
    print("="*80)

    for i, listing in enumerate(sorted_listings, start=1):
        print(f"{i}. [{listing['model']}] - ${listing['price']:.2f}")
        print(f"   Title: {listing['title']}")
        print(f"   Link:  {listing['url']} (via {listing['engine']})")
        print("-" * 80)

if __name__ == "__main__":
    main()
