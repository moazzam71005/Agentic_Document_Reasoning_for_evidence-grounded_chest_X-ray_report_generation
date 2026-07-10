"""
Unified LLM client — supports free cloud APIs (no local GPU required).

Providers (set LLM_PROVIDER in .env):
  groq       — FREE, fast, recommended (https://console.groq.com)
  gemini     — FREE tier via Google AI Studio (https://aistudio.google.com)
  huggingface — FREE Inference API, rate-limited
  ollama     — local only, needs GPU for 8B+ models
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
        self.model = model or self._default_model()
        self._validate_config()

    def _default_model(self) -> str:
        defaults = {
            "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "gemini": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            "huggingface": os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            "ollama": os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        }
        return defaults.get(self.provider, "llama-3.3-70b-versatile")

    def _validate_config(self) -> None:
        key_map = {
            "groq": "GROQ_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "huggingface": "HF_API_KEY",
        }
        if self.provider in key_map and not os.getenv(key_map[self.provider]):
            raise ValueError(
                f"{key_map[self.provider]} not set. "
                f"Copy .env.example to .env and add your free API key."
            )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        json_mode: bool = False,
    ) -> str:
        dispatch = {
            "groq": self._chat_groq,
            "gemini": self._chat_gemini,
            "huggingface": self._chat_huggingface,
            "ollama": self._chat_ollama,
        }
        fn = dispatch.get(self.provider)
        if fn is None:
            raise ValueError(f"Unknown provider: {self.provider}")
        return fn(messages, temperature, top_p, json_mode)

    def _chat_groq(
        self, messages, temperature, top_p, json_mode
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _chat_gemini(
        self, messages, temperature, top_p, json_mode
    ) -> str:
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(self.model)

        # Flatten to single prompt for simplicity
        parts = []
        for m in messages:
            role = m["role"].upper()
            parts.append(f"[{role}]\n{m['content']}")
        prompt = "\n\n".join(parts)
        if json_mode:
            prompt += "\n\nRespond with valid JSON only."

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
            ),
        )
        return response.text

    def _chat_huggingface(
        self, messages, temperature, top_p, json_mode
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 2048,
        }
        resp = httpx.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['HF_API_KEY']}"},
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _chat_ollama(
        self, messages, temperature, top_p, json_mode
    ) -> str:
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "top_p": top_p},
        }
        if json_mode:
            payload["format"] = "json"

        resp = httpx.post(f"{base}/api/chat", json=payload, timeout=300.0)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, tolerating markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)
