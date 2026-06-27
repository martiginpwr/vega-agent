from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.ollama import OllamaError, ollama_client
from backend.app.schemas import ChatMessage, ChatRequest, ChatResponse, HealthResponse, ModelsResponse

LOCAL_SYSTEM_PROMPT = """You are Vega Agent, a private local-first personal AI assistant.
You run through local Ollama models only. Be concise, practical, and transparent about uncertainty.
Never claim to use cloud services. If a requested capability is not implemented yet, say so briefly."""

app = FastAPI(title="Vega Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    connected = await ollama_client.health()
    return HealthResponse(
        ok=True,
        ollama_url=settings.ollama_base_url,
        ollama_connected=connected,
    )


@app.get("/api/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    try:
        local_models = await ollama_client.list_models()
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    default_model = settings.vega_default_model or (local_models[0].name if local_models else None)
    return ModelsResponse(models=local_models, default_model=default_model)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    messages = request.messages
    if not messages or messages[0].role != "system":
        messages = [ChatMessage(role="system", content=LOCAL_SYSTEM_PROMPT), *messages]

    try:
        assistant_message = await ollama_client.chat(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
        )
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatResponse(model=request.model, message=assistant_message)
