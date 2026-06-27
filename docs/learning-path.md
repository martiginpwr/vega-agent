# Learning Path

This project is also a practical AI agent engineering course. Each milestone should teach one set of concepts and leave behind readable documentation.

## 1. Local Model Serving

Learn how Ollama exposes local models, how chat payloads are structured, how model selection works, and how latency changes across model sizes.

Deliverable: a local chat UI that can switch models.

## 2. State And Sessions

Learn the difference between raw transcripts, active context, and durable state.

Deliverable: SQLite-backed conversations that can be resumed.

## 3. Memory

Learn how to decide what should become memory, how to store it with provenance, and how to retrieve it without polluting every prompt.

Deliverable: inspectable local memories with edit and delete controls.

## 4. Retrieval

Learn chunking, embeddings, vector search, metadata filters, and retrieval evaluation.

Deliverable: local document chat with visible retrieved sources.

## 5. Agent Loops

Learn the observe, decide, act, observe loop. Compare LLM-driven and code-driven orchestration.

Deliverable: a typed agent run trace that can include tool calls.

## 6. Tool Safety

Learn schema validation, permission levels, human approval, failure handling, and audit trails.

Deliverable: a small local tool registry with safe defaults.

## 7. Context Compaction

Learn why long context is not the same as good context. Practice handoff summaries, state compression, and recovery from compacted history.

Deliverable: manual and automatic compaction for long sessions.

## 8. Routing

Learn when not to call the main model. Use rules, embeddings, and small local classifiers for cheap decisions.

Deliverable: local router that can choose chat, memory, retrieval, or tool paths.

## 9. Skills

Learn how successful workflows become reusable artifacts.

Deliverable: reviewed local skill drafts generated from completed sessions.
