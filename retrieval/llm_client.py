import json
import re

import numpy as np
import requests


class LLMClientError(RuntimeError):
    """Base exception for generation/embedding client failures."""


class OllamaError(LLMClientError):
    """Backward-compatible Ollama exception used by existing modules."""


class OpenAICompatibleError(LLMClientError):
    """Exception raised by OpenAI-compatible chat endpoints."""

class SentenceTransformerEmbeddingClient:
    """
    Local embedding client backed by sentence-transformers.

    It intentionally exposes the same `.embed(...)` interface used by
    HybridRetriever, so the retrieval architecture does not need to know
    whether embeddings come from Ollama or Hugging Face.
    """

    def __init__(
        self,
        model_name,
        device="auto",
        query_prefix="",
        document_prefix="",
    ):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise LLMClientError(
                "缺少 sentence-transformers。請先執行："
                "pip install -U sentence-transformers"
            ) from exc

        if str(device).strip().lower() == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = str(model_name)
        self.device = str(device)

        self.query_prefix = str(query_prefix or "")
        self.document_prefix = str(document_prefix or "")

        print(
            f"載入 SentenceTransformer embedding model："
            f"{self.model_name} ({self.device})"
        )

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

    def embed(self, model, texts, input_type="passage"):
        if str(model) != self.model_name:
            raise LLMClientError(
                f"Embedding model 不一致："
                f"client={self.model_name!r}, requested={model!r}"
            )

        if isinstance(texts, str):
            texts = [texts]

        input_type = str(input_type or "").strip().lower()

        if input_type == "query":
            prefix = self.query_prefix
        elif input_type == "passage":
            prefix = self.document_prefix
        else:
            prefix = ""

        prepared_texts = [
            f"{prefix}{str(text)}"
            for text in texts
        ]

        vectors = self.model.encode(
            prepared_texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

        return np.asarray(vectors, dtype=np.float32)

def _parse_json_content(content):
    """Parse JSON returned as plain text or inside a Markdown code fence."""
    if isinstance(content, (dict, list)):
        return content

    text = str(content or "").strip()
    if not text:
        raise json.JSONDecodeError("empty content", text, 0)

    # Common model output: ```json\n{...}\n```
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback for models that prepend/append a short explanation despite instructions.
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
            return value
        except json.JSONDecodeError:
            continue

    raise json.JSONDecodeError("no valid JSON object found", text, 0)


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
            return _parse_json_content(content)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Structured output 無法解析：{str(content)[:1500]}") from exc

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

    def embed(self, model, texts, input_type=None):
        payload = {"model": model, "input": texts}
        data = self._post("/api/embed", payload)

        embeddings = data.get("embeddings")
        if not embeddings:
            raise OllamaError("Ollama /api/embed 沒有回傳 embeddings")

        return np.asarray(embeddings, dtype=np.float32)


class OpenAICompatibleClient:
    """Client for OpenAI-compatible /v1/chat/completions endpoints.

    The public method signatures intentionally match OllamaClient so existing
    Router/Summarizer/Evidence/Answer modules can switch clients without changes.
    """

    def __init__(
        self,
        base_url,
        timeout=180,
        api_key=None,
        headers=None,
        verify_ssl=True,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = bool(verify_ssl)
        self.session = requests.Session()

        default_headers = {"Content-Type": "application/json"}
        if api_key:
            default_headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            default_headers.update(headers)
        self.session.headers.update(default_headers)

    def _post(self, endpoint, payload):
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
        except requests.RequestException as exc:
            raise OpenAICompatibleError(f"無法連線到 OpenAI-compatible API：{url}\n{exc}") from exc

        if response.status_code >= 400:
            raise OpenAICompatibleError(
                f"OpenAI-compatible API 錯誤 {response.status_code}：{response.text[:2000]}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise OpenAICompatibleError(
                f"OpenAI-compatible API 回傳非 JSON：{response.text[:1000]}"
            ) from exc

    def _message_content(self, data):
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenAICompatibleError(
                "OpenAI-compatible API 缺少 choices[0].message.content："
                f"{str(data)[:1500]}"
            ) from exc

        if isinstance(content, str):
            return content.strip()

        # Some OpenAI-compatible servers may return content parts.
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                elif item:
                    parts.append(str(item))
            return "\n".join(parts).strip()

        return str(content or "").strip()

    def chat_text(self, model, messages, temperature=0.2, think=True):
        # `think` is accepted for interface compatibility. It is intentionally
        # not sent because it is an Ollama-specific request field.
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        data = self._post("/chat/completions", payload)
        return self._message_content(data)

    def chat_json(self, model, messages, schema, temperature=0.0, think=True):
        # Do not depend on response_format/json_schema support because not every
        # OpenAI-compatible server implements it. Give the schema directly to
        # the model and robustly parse the returned text instead.
        schema_text = json.dumps(schema, ensure_ascii=False)
        json_instruction = (
            "你必須只輸出一個符合下列 JSON Schema 的合法 JSON 值。"
            "不要輸出 Markdown code fence、解釋、前言或結語。\n"
            f"JSON Schema：\n{schema_text}"
        )

        constrained_messages = [
            {"role": "system", "content": json_instruction},
            *messages,
        ]

        content = self.chat_text(
            model,
            constrained_messages,
            temperature=temperature,
            think=think,
        )

        try:
            return _parse_json_content(content)
        except json.JSONDecodeError as exc:
            raise OpenAICompatibleError(
                f"Structured output 無法解析：{content[:1500]}"
            ) from exc

    def embed(self, model, texts, input_type=None):
        raise OpenAICompatibleError(
            "此 OpenAICompatibleClient 目前只設定 chat/completions；"
            "embedding 請繼續使用 OllamaClient。"
        )
