import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def ollama_generate_json(
    prompt: str,
    model: str = "llama3.1",
    timeout: int = 120,
) -> dict:
    """
    Sends a prompt to Ollama and expects the response to be valid JSON.
    Raises ValueError if parsing fails.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 700,   # prevent ultra-long rambles
        },
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    text = r.json().get("response", "").strip()

    # Some models wrap JSON in text; try to extract the first JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Ollama did not return JSON. Raw:\n{text}")

    json_str = text[start:end+1]
    return json.loads(json_str)
