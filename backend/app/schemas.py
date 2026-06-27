from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0, le=2)


class ChatResponse(BaseModel):
    model: str
    message: ChatMessage
    done: bool = True


class ModelInfo(BaseModel):
    name: str
    size: int | None = None
    modified_at: str | None = None
    capabilities: list[str] = Field(default_factory=list)


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    default_model: str | None = None


class HealthResponse(BaseModel):
    ok: bool
    ollama_url: str
    ollama_connected: bool
