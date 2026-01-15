import json
import os
from typing import Any, Dict
from litellm import completion

# You can replace these with other models as needed but this is the one we suggest for this lab.
MODEL = "groq/llama-3.3-70b-versatile"

def get_api_key() -> str:
   api_key = os.getenv("GROQ_API_KEY")
   if not api_key:
      raise RuntimeError("API Key not set")
   return api_key


def get_itinerary(destination: str) -> Dict[str, Any]:
    """
    Returns a JSON-like dict with keys:
      - destination
      - price_range
      - ideal_visit_times
      - top_attractions
    """
    # implement litellm call here to generate a structured travel itinerary for the given destination

    # See https://docs.litellm.ai/docs/ for reference.

    api_key = get_api_key()

    prompt = f"""
Return ONLY valid JSON with this exact schema:
{{
  "destination": "{destination}",
  "price_range": "budget | mid-range | luxury",
  "ideal_visit_times": ["..."],
  "top_attractions": ["..."]
}}

Rules:
- JSON only (no markdown)
- ideal_visit_times must be an array of strings
- top_attractions must be an array of strings

"""

    response = completion(
       model=MODEL,
       messages=[{"role": "user", "content": prompt}],
       api_key=api_key,
       response_format={"type": "json_object"},
       max_tokens=300
    )

    content = response["choices"][0]["message"]["content"]
    data = json.loads(content)

    # schema validation
    required = {"destination", "price_range", "ideal_visit_times", "top_attractions"}
    if not required.issubset(data.keys()):
       missing = required - set(data.keys())
       raise ValueError(f"Missing keys: {missing}")
    
    if not isinstance(data["ideal_visit_times"], list) or not all(isinstance(x, str) for x in data["ideal_visit_times"]):
       raise ValueError("ideal_visit_times must be a list of strings")
    
    if not isinstance(data["top_attractions"], list) or not all(isinstance(x, str) for x in data["top_attractions"]):
       raise ValueError("top_attractions must be a list of strings")

    return data
