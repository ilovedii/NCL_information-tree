import json

import numpy as np
import requests


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url, timeout=180):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _post(self, endpoint, payload):
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as exc:
            raise OllamaError(f"無法連線到 Ollama：{url}\n{exc}") from exc
        if response.status_code >= 400:
            raise OllamaError(f"Ollama API 錯誤 {response.status_code}：{response.text}")
        try:
            return response.json()
        except ValueError as exc:
            raise OllamaError(f"Ollama 回傳非 JSON：{response.text[:1000]}") from exc

    def chat_json(self, model, messages, schema, temperature=0.0, think=True):
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "think": think,
            "options": {"temperature": temperature},
        }
        data = self._post("/api/chat", payload)
        content = data.get("message", {}).get("content", "")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Structured output 無法解析：{content[:1500]}") from exc

    def chat_text(self, model, messages, temperature=0.2, think=True):
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": {"temperature": temperature},
        }
        data = self._post("/api/chat", payload)
        return data.get("message", {}).get("content", "").strip()

    def embed(self, model, texts):
        payload = {"model": model, "input": texts}
        data = self._post("/api/embed", payload)
        embeddings = data.get("embeddings")
        if not embeddings:
            raise OllamaError("Ollama /api/embed 沒有回傳 embeddings")
        return np.asarray(embeddings, dtype=np.float32)