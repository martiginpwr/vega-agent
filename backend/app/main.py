from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.db import database
from backend.app.memory import build_memory_context_message, classify_memory_for_conversation, retrieve_relevant_memories
from backend.app.ollama import OllamaError, ollama_client
from backend.app.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationSummary,
    CreateConversationRequest,
    HealthResponse,
    MemoryRecord,
    ModelsResponse,
    StoredMessage,
    TraceResponse,
)

LOCAL_SYSTEM_PROMPT = """You are Vega Agent, a private local-first personal AI assistant. You are fluent in English and Russian."""

app = FastAPI(title="Vega Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def is_chat_model(model):
    return "completion" in model.capabilities or not model.capabilities


def is_background_only_model(model_name: str) -> bool:
    background_fragments = {
        "qwen2.5-0.5b-instruct",
    }
    normalized_name = model_name.lower()
    return any(fragment in normalized_name for fragment in background_fragments)


def choose_default_model(models):
    for model in models:
        if "completion" in model.capabilities or not model.capabilities:
            return model.name
    return models[0].name if models else None


def is_internal_model(model_name: str, chat_model_names: set[str]) -> bool:
    internal_models = {
        settings.vega_memory_model,
        settings.vega_memory_verifier_model,
    }
    if settings.vega_memory_grounding_model in internal_models:
        internal_models.add(settings.vega_memory_grounding_model)
    shared_with_chat = {
        settings.vega_default_model,
        "qwen3.5:9b",
        settings.vega_memory_grounding_model,
    }
    internal_models -= {model for model in shared_with_chat if model in chat_model_names}
    return model_name in internal_models


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

    chat_model_names = {model.name for model in local_models if is_chat_model(model)}
    chat_models = [
        model
        for model in local_models
        if (
            is_chat_model(model)
            and not is_internal_model(model.name, chat_model_names)
            and not is_background_only_model(model.name)
        )
    ]
    default_model = settings.vega_default_model or choose_default_model(chat_models)
    if default_model and default_model not in {model.name for model in chat_models}:
        default_model = choose_default_model(chat_models)

    return ModelsResponse(models=chat_models, default_model=default_model)


@app.get("/api/conversations", response_model=list[ConversationSummary])
async def list_conversations() -> list[ConversationSummary]:
    return [ConversationSummary(**conversation) for conversation in database.list_conversations()]


@app.post("/api/conversations", response_model=ConversationSummary)
async def create_conversation(request: CreateConversationRequest) -> ConversationSummary:
    conversation = database.create_conversation(
        title=request.title or "New chat",
        selected_model=request.selected_model,
    )
    return ConversationSummary(**conversation)


@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str) -> ConversationDetail:
    try:
        conversation = database.get_conversation(conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found.") from exc

    messages = database.list_messages(conversation_id)
    return ConversationDetail(
        conversation=ConversationSummary(**conversation),
        messages=[StoredMessage(**message) for message in messages],
    )


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str) -> dict[str, bool]:
    try:
        database.get_conversation(conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found.") from exc

    database.delete_conversation(conversation_id)
    return {"ok": True}


@app.get("/api/memories", response_model=list[MemoryRecord])
async def list_memories() -> list[MemoryRecord]:
    return [MemoryRecord(**memory) for memory in database.list_memories()]


@app.get("/api/traces/{run_id}", response_model=TraceResponse)
async def get_trace(run_id: str) -> TraceResponse:
    try:
        trace = database.get_trace(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Trace not found.") from exc
    return TraceResponse(**trace)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    try:
        local_models = await ollama_client.list_models()
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    chat_model_names = {model.name for model in local_models if is_chat_model(model)}
    if request.model not in chat_model_names:
        raise HTTPException(status_code=400, detail="Selected model is not a chat-capable Ollama model.")

    user_messages = [message for message in request.messages if message.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="At least one user message is required.")

    latest_user_message = user_messages[-1]
    if request.conversation_id:
        try:
            conversation = database.get_conversation(request.conversation_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Conversation not found.") from exc
    else:
        title = latest_user_message.content.strip().replace("\n", " ")
        conversation = database.create_conversation(
            title=title[:46] if title else "New chat",
            selected_model=request.model,
        )

    conversation_id = conversation["id"]
    run = database.create_agent_run(conversation_id=conversation_id, model=request.model)
    run_id = run["id"]
    database.add_trace_event(
        run_id=run_id,
        step="chat.start",
        status="completed",
        message="Started local chat run.",
        metadata={
            "model": request.model,
            "conversation_id": conversation_id,
            "think": request.think,
        },
    )
    database.update_conversation_model(conversation_id, request.model)
    user_row = database.add_message(
        conversation_id=conversation_id,
        role="user",
        content=latest_user_message.content,
        model=request.model,
        metadata={"run_id": run_id},
    )
    database.update_agent_run(run_id, status="started", user_message_id=user_row["id"])
    database.add_trace_event(
        run_id=run_id,
        step="chat.persist_user",
        status="completed",
        message="Saved user message to SQLite.",
        metadata={"message_id": user_row["id"]},
    )
    database.maybe_title_from_message(conversation_id, latest_user_message.content)

    retrieved_memories = []
    try:
        retrieved_memories = await retrieve_relevant_memories(
            query=latest_user_message.content,
            run_id=run_id,
        )
    except OllamaError as exc:
        database.add_trace_event(
            run_id=run_id,
            step="memory.retrieve",
            status="failed",
            message="Memory retrieval failed; continuing chat without retrieved memory.",
            metadata={"error": str(exc)},
        )

    messages = request.messages
    memory_context_message = build_memory_context_message(retrieved_memories)
    if not messages or messages[0].role != "system":
        messages = [ChatMessage(role="system", content=LOCAL_SYSTEM_PROMPT), *messages]
    if memory_context_message:
        messages = [messages[0], memory_context_message, *messages[1:]]

    try:
        database.add_trace_event(
            run_id=run_id,
            step="model.chat",
            status="started",
            message="Sending conversation context to local Ollama chat model.",
            metadata={"message_count": len(messages), "model": request.model},
        )
        assistant_message = await ollama_client.chat(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            think=request.think,
        )
    except OllamaError as exc:
        database.add_trace_event(
            run_id=run_id,
            step="model.chat",
            status="failed",
            message="Local Ollama chat model request failed.",
            metadata={"error": str(exc)},
        )
        database.update_agent_run(run_id, status="failed", error=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    database.add_trace_event(
        run_id=run_id,
        step="model.chat",
        status="completed",
        message="Received assistant response from local Ollama chat model.",
        metadata={"response_chars": len(assistant_message.content)},
    )

    assistant_row = database.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=assistant_message.content,
        model=request.model,
        metadata={"responding_to": user_row["id"], "run_id": run_id},
    )
    database.update_agent_run(
        run_id,
        status="completed",
        assistant_message_id=assistant_row["id"],
    )
    database.add_trace_event(
        run_id=run_id,
        step="chat.persist_assistant",
        status="completed",
        message="Saved assistant response to SQLite.",
        metadata={"message_id": assistant_row["id"]},
    )

    job_id = database.create_memory_job(conversation_id)
    database.add_trace_event(
        run_id=run_id,
        step="memory.queue",
        status="completed",
        message="Queued automatic memory extraction and verification job.",
        metadata={
            "job_id": job_id,
            "classifier_model": settings.vega_memory_model,
            "grounding_model": settings.vega_memory_grounding_model,
            "verifier_model": settings.vega_memory_verifier_model,
            "embedding_model": settings.vega_embedding_model,
        },
    )
    background_tasks.add_task(classify_memory_for_conversation, conversation_id, job_id, run_id)

    return ChatResponse(
        model=request.model,
        message=assistant_message,
        conversation_id=assistant_row["conversation_id"],
        run_id=run_id,
        user_message_id=user_row["id"],
        assistant_message_id=assistant_row["id"],
    )
