# ollama_client.py
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

import requests


def _base_url() -> str:
    """
    Default to 127.0.0.1 (more reliable than localhost).
    Allow override via OLLAMA_HOST, e.g.
      export OLLAMA_HOST="http://host.docker.internal:11434"
    """
    return os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def ollama_is_running(timeout: int = 3) -> bool:
    try:
        r = requests.get(f"{_base_url()}/api/tags", timeout=timeout)
        return r.ok
    except Exception:
        return False


def ollama_list_models(timeout: int = 5) -> Dict[str, Any]:
    r = requests.get(f"{_base_url()}/api/tags", timeout=timeout)
    r.raise_for_status()
    return r.json()


def _extract_first_json_object(text: str) -> str:
    """
    Extract the first {...} JSON object from a model response that may include extra text.
    """
    text = text.strip()
    # Fast path: already JSON
    if text.startswith("{") and text.endswith("}"):
        return text

    # Find first balanced JSON object by scanning braces
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found. Raw:\n{text}")

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError(f"Unbalanced JSON braces. Raw:\n{text}")


def ollama_generate_json(
    prompt: str,
    model: str = "llama3.1",
    timeout: int = 120,
    temperature: float = 0.2,
    num_predict: int = 900,
    max_retries: int = 1,
) -> Dict[str, Any]:
    """
    Send a prompt to Ollama and return parsed JSON dict.
    Retries once with a "fix to valid JSON" instruction if parsing fails.
    """
    url = f"{_base_url()}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    last_err: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            raw = r.json().get("response", "").strip()

            json_str = _extract_first_json_object(raw)
            return json.loads(json_str)

        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                break

            # Retry with explicit repair instruction
            payload["prompt"] = (
                prompt
                + "\n\nIMPORTANT: Your previous output was not valid JSON. "
                  "Return ONLY valid JSON (no markdown, no commentary), matching the schema exactly."
            )

    raise RuntimeError(f"Ollama JSON generation failed: {last_err}")
