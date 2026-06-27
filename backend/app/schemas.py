from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    conversation_id: str | None = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    think: bool | None = None


class ChatResponse(BaseModel):
    model: str
    message: ChatMessage
    conversation_id: str
    run_id: str
    user_message_id: str
    assistant_message_id: str
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


class StoredMessage(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    model: str | None = None
    created_at: str
    metadata: dict = Field(default_factory=dict)


class ConversationSummary(BaseModel):
    id: str
    title: str
    selected_model: str | None = None
    created_at: str
    updated_at: str
    message_count: int = 0


class ConversationDetail(BaseModel):
    conversation: ConversationSummary
    messages: list[StoredMessage]


class CreateConversationRequest(BaseModel):
    title: str | None = None
    selected_model: str | None = None


class MemoryRecord(BaseModel):
    id: str
    type: str
    content: str
    status: str
    confidence: float | None = None
    importance: float | None = None
    rationale: str | None = None
    source_conversation_id: str
    created_at: str


class TraceEvent(BaseModel):
    id: str
    run_id: str
    step: str
    status: str
    message: str
    metadata: dict = Field(default_factory=dict)
    created_at: str


class AgentRun(BaseModel):
    id: str
    conversation_id: str
    user_message_id: str | None = None
    assistant_message_id: str | None = None
    model: str | None = None
    status: str
    error: str | None = None
    created_at: str
    completed_at: str | None = None


class TraceResponse(BaseModel):
    run: AgentRun
    events: list[TraceEvent]
