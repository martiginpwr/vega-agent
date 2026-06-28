# Architecture

Vega Agent is designed as a local-first agent workbench. The current implementation is the smallest useful vertical slice: browser UI, local backend, Ollama model listing, model selection, and chat.

## Current System

```text
Browser UI (React/Vite)
        |
        | HTTP /api/*
        v
FastAPI backend
        |
        | HTTP localhost:11434
        v
Ollama local models
```

The frontend is a local web app. It never talks to Ollama directly; it calls the Vega backend. That keeps provider details, future memory retrieval, tool execution, and safety checks in one local runtime.

The backend currently exposes:

- `GET /api/health`: checks whether Ollama is reachable.
- `GET /api/models`: returns locally installed Ollama models.
- `POST /api/chat`: sends a conversation to the selected Ollama model.

## Planned Agent Runtime

The agent runtime should eventually sit between `/api/chat` and the model provider.

```text
User message
  -> Session state loader
  -> Memory retriever
  -> RAG retriever
  -> Context builder
  -> Router
  -> Main model or smaller local component
  -> Tool planner
  -> Tool executor with policy checks
  -> Verifier
  -> Response writer
  -> Memory and skill candidate evaluator
```

This should be implemented incrementally. The first serious internal abstraction should be an `AgentRun` state object that records the user input, selected model, retrieved context, tool calls, observations, final answer, and errors.

## Memory Design

Vega should use three different concepts instead of one overloaded memory bucket:

- Conversation history: complete local session transcripts.
- Working context: the subset of history and retrieved data used in the current model call.
- Long-term memory: durable facts, preferences, project notes, and reusable lessons.

Memory creation should be automatic and classifier-driven. The system should not use simple keyword rules as the primary signal because natural conversations are too creative and varied. After each conversation turn is saved, a background local classifier/extractor should decide whether the exchange contains memory-worthy information, what type of memory it is, and how confident the system should be.

The implementation isolates memory decisions behind local model roles instead of using the main chat model for every background judgment. The same small local model can serve multiple roles at first, but each role has a separate prompt and trace step so it can be replaced independently later.

Memory storage should include provenance: when it was created, from which session, which messages caused it, why it was considered useful, and when it was last used. This makes deletion, inspection, correction, deduplication, and conflict handling possible.

Initial memory types:

- `preference`: user style or behavior preferences.
- `identity`: stable user details.
- `project`: durable project constraints and decisions.
- `fact`: stable factual information the user provided.
- `procedure`: recurring workflow instructions.
- `correction`: corrections to Vega's assumptions or behavior.
- `skill_candidate`: patterns that might become reusable workflows later.

Memory pipeline:

```text
messages persisted
  -> background memory job
  -> local classifier/extractor
  -> candidate memories with type, confidence, importance, and rationale
  -> local grounding verifier against user-authored source messages
  -> optional local embedding duplicate check against similar memories
  -> local duplicate/conflict verifier
  -> active or suggested memory record
  -> future retrieval into working context
```

The initial implementation is intentionally conservative:

- SQLite stores every conversation and message.
- `VEGA_MEMORY_MODEL` controls the local Ollama model used for memory classification. This defaults to the stronger local chat model because the 0.5B extractor produced too many hallucinated candidates.
- `VEGA_MEMORY_GROUNDING_MODEL` controls the local Ollama model used to verify that a candidate is supported by user-authored messages. This defaults to the stronger local chat model because false memory writes are more expensive than extra latency.
- `VEGA_MEMORY_VERIFIER_MODEL` controls the local Ollama model used for duplicate, conflict, and final save/reject decisions. This also defaults to the stronger local chat model for now.
- `VEGA_EMBEDDING_MODEL` optionally enables local embedding-based deduplication before storing a candidate.
- The grounding verifier is intentionally separate from the extractor. The extractor proposes; the grounding verifier rejects unsupported claims, cites exact user quotes, and returns the supported memory content that will be stored.

## Retrieval Design

The first RAG implementation should be simple:

1. Store documents locally.
2. Chunk them deterministically.
3. Embed chunks with a local embedding model.
4. Persist vectors and metadata on disk.
5. Retrieve top matches for the current query.
6. Add concise citations into the context package.

Likely first choices:

- Chroma for a simple local persistent vector store.
- Ollama embeddings with `all-minilm`, `embeddinggemma`, or `qwen3-embedding`.
- Sentence Transformers as an alternate local embedding path.

## Context Compaction

Compaction should produce a handoff summary, not a vague recap. A useful compacted state should include:

- User goal.
- Current plan.
- Decisions already made.
- Files, memories, or documents already used.
- Tool results that still matter.
- Open questions.
- Constraints and preferences.

For local models, compaction should be conservative. The system can keep raw history on disk and compact only the working context sent to the model.

## Tool Use

Tools should be registered with typed schemas and permission metadata:

- `read_only`: safe inspection tools.
- `local_write`: tools that can change local files or memory.
- `exec`: tools that run commands or scripts.
- `external`: disabled by default because Vega is local-first.

The first tool layer should support read-only local knowledge tools before adding file writes or shell execution.

## Skill Creation

The Hermes-style skill idea should be implemented as a review pipeline:

1. After a successful session, a local evaluator asks whether a reusable pattern exists.
2. If yes, it writes a candidate skill draft with trigger conditions, steps, examples, and required tools.
3. The user reviews and accepts or rejects it.
4. Accepted skills become local files.
5. The router can retrieve relevant skills for future tasks.

Automatic skill creation should stay opt-in until the evaluator is reliable.
