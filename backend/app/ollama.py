import httpx

from backend.app.config import settings
from backend.app.schemas import ChatMessage, ModelInfo


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                response = await client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[ModelInfo]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not reach Ollama at {self.base_url}") from exc

        payload = response.json()
        return [
            ModelInfo(
                name=model.get("name", ""),
                size=model.get("size"),
                modified_at=model.get("modified_at"),
                capabilities=model.get("capabilities", []),
            )
            for model in payload.get("models", [])
            if model.get("name")
        ]

    async def chat(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        temperature: float,
        think: bool | None = None,
        response_format: str | None = None,
    ) -> ChatMessage:
        payload = {
            "model": model,
            "messages": [message.model_dump() for message in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if think is not None:
            payload["think"] = think
        if response_format is not None:
            payload["format"] = response_format

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise OllamaError(f"Ollama rejected the chat request: {detail}") from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not reach Ollama at {self.base_url}") from exc

        data = response.json()
        message = data.get("message") or {}
        content = message.get("content", "").strip()
        if not content:
            raise OllamaError("Ollama returned an empty assistant message.")
        return ChatMessage(role="assistant", content=content)

    async def embed(self, *, model: str, text: str) -> list[float]:
        payload = {"model": model, "input": text}

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(f"{self.base_url}/api/embed", json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise OllamaError(f"Ollama rejected the embedding request: {detail}") from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not reach Ollama at {self.base_url}") from exc

        data = response.json()
        embeddings = data.get("embeddings") or []
        if embeddings and isinstance(embeddings[0], list):
            return [float(value) for value in embeddings[0]]
        embedding = data.get("embedding") or []
        if embedding:
            return [float(value) for value in embedding]
        raise OllamaError("Ollama returned no embedding vector.")


ollama_client = OllamaClient(settings.ollama_base_url)
