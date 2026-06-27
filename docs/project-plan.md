# Vega Agent Project Plan

This document is the high-level map for Vega Agent. The goal is to build a useful local personal AI agent while using the project as a practical learning path for agent engineering.

## North Star

Vega is a local-first agent workbench:

- The browser UI feels as simple as ChatGPT for everyday use.
- The backend owns the agent loop, memory, retrieval, tools, routing, and safety checks.
- Ollama is the first local model provider.
- All state, embeddings, vector indexes, memories, skills, and tools stay on the machine.
- The architecture should remain modular enough to add new local models and capabilities without rewriting the app.

## Design Principles

- Start with one useful vertical slice, then deepen it.
- Keep the visible UI quiet. Advanced state belongs behind drawers, modals, traces, or developer views.
- Treat the transcript, working context, long-term memory, retrieved documents, and generated skills as different artifacts.
- Use the strongest local model for broad reasoning and smaller local models or deterministic rules for cheap classification, routing, and memory candidacy.
- Every automated action should leave a local audit trail the user can inspect.
- Prefer human approval for durable memory, file writes, shell execution, and reusable skills until the system has earned trust.

## Research-Informed Architecture

Modern coding and agent systems tend to separate a few concerns:

- Context assembly: decide what the model sees now.
- Planning: break down multi-step work when needed.
- Tool selection: call typed tools when they are useful, not for every turn.
- Execution: run tools under a permission model.
- Verification: check whether the answer or action actually worked.
- Persistence: store raw history, summaries, memory, traces, and reusable patterns separately.

Vega will adapt that pattern locally:

```text
User message
  -> session loader
  -> memory and document retrieval
  -> context builder
  -> router
  -> local model call
  -> optional tool loop
  -> verifier
  -> response
  -> memory / skill candidate review
```

## Development Phases

### Phase 0: Local Chat Foundation

Status: in progress.

Purpose: prove the local-only path works end to end.

Features:

- React web chat interface.
- FastAPI backend.
- Ollama health check.
- Local model listing.
- Per-request model selection.
- Basic docs and project structure.

How it works:

- The browser calls the Vega backend at `/api/*`.
- The backend calls Ollama at `127.0.0.1:11434`.
- The frontend never calls cloud APIs or Ollama directly.

### Phase 1: Conversation Persistence

Purpose: make the chat usable across sessions.

Features:

- SQLite database under local project storage.
- Conversations, messages, model metadata, and timestamps.
- Sidebar sessions: create, rename, delete, resume.
- Export/import for local backups.

Design choice:

- Raw transcript history is stored separately from the context sent to the model. That lets Vega keep everything while sending only the useful subset.

### Phase 2: Memory

Purpose: let Vega remember stable facts and preferences without stuffing every chat into context.

Features:

- Manual "save memory" action first.
- Memory editor with delete/disable.
- Memory metadata: source message, reason, confidence, created date, last used date.
- Retrieval of relevant memories into future chats.

Later:

- Local memory-candidate classifier.
- Human approval queue for suggested memories.

### Phase 3: Local RAG

Purpose: let Vega answer from local documents.

Features:

- Local document ingestion.
- Deterministic chunking.
- Local embeddings through Ollama or Sentence Transformers.
- Local vector store, likely Chroma at first.
- Citations surfaced in a details drawer.

Hardware note:

- Embeddings should use a small local embedding model, not the main chat model. Your Qwen embedding model is a realistic candidate.

### Phase 4: Context Compaction

Purpose: handle long sessions with limited local model context.

Features:

- Token/context budget tracking per selected model.
- Manual compaction.
- Automatic compaction near budget thresholds.
- Stored handoff summaries.

A good summary includes:

- Current goal.
- Decisions made.
- Constraints.
- Files or docs already used.
- Tool results that still matter.
- Open questions.

### Phase 5: Tool Runtime

Purpose: let Vega do useful local actions safely.

Features:

- Typed tool registry.
- Tool permission classes: read-only, local write, execute, external-disabled.
- Tool traces in the UI.
- User approval for risky tools.

Initial tools:

- Search local docs.
- Read local files within approved folders.
- Inspect project structure.

Later tools:

- File edits.
- Shell commands.
- Git operations.

### Phase 6: Routing And Small Local Models

Purpose: avoid using a 7B-14B chat model for every tiny decision.

Features:

- Deterministic routing rules first.
- Lightweight local classifiers for intent, memory candidacy, and retrieval need.
- Latency and quality comparisons.

Realistic local strategy:

- Main chat/reasoning: Qwen, Mistral, or similar Ollama chat models.
- Embeddings: small embedding model.
- Classification: small BERT-style model, sentence-transformer classifier, or a very small Ollama model after benchmarking.

### Phase 7: Reusable Skills

Purpose: explore the Hermes-style idea of converting successful conversations into reusable local workflows.

Features:

- Skill artifact format stored locally as Markdown or JSON.
- Post-session skill candidate evaluator.
- Human review before a skill becomes active.
- Skill retrieval during future context assembly.

Skill creation pipeline:

```text
Successful conversation
  -> evaluator asks "was there a reusable pattern?"
  -> candidate skill draft
  -> user review
  -> accepted local skill
  -> future retrieval by trigger
```

Important constraint:

- Automatic skill creation should be conservative. A bad skill can pollute future behavior, so accepted skills need provenance, versioning, and a way to disable them.

### Phase 8: Evaluation Harness

Purpose: keep the agent from getting worse as features are added.

Features:

- Scripted local eval tasks.
- Latency measurements.
- Retrieval quality checks.
- Memory precision checks.
- Tool-call success checks.
- UI smoke tests with Playwright.

## Hardware Strategy

Current target:

- RTX 4070 with 12 GB VRAM.
- 16 GB RAM, possible 32 GB later.
- 16-core CPU.

Practical implications:

- Keep one main chat model loaded when possible.
- Do not use the main model for every router or classifier step.
- Prefer local SQLite and lightweight vector storage before heavier infrastructure.
- Add streaming responses soon so slow local generations still feel usable.
- Benchmark each model role before making it default.

## Near-Term Build Order

1. Simplify and verify the chat UI.
2. Add SQLite conversation persistence.
3. Add streaming Ollama responses.
4. Add local memory storage and manual memory controls.
5. Add local embeddings and document retrieval.
6. Add context compaction.
7. Add a typed tool registry with read-only tools.
8. Add routing and classifier experiments.
9. Add skill-candidate generation behind human approval.
10. Add local evals and regression tests.

## Documentation Plan

The docs should grow with the code:

- `docs/project-plan.md`: where the project is headed.
- `docs/architecture.md`: how the system is structured now.
- `docs/research.md`: source-backed research notes.
- `docs/learning-path.md`: concepts to learn by milestone.
- Future `docs/components/*.md`: deep dives for memory, RAG, tools, routing, and skills.

Each major feature should explain:

- What problem it solves.
- How it works.
- Why this design was chosen.
- What local model or storage component it uses.
- How it can fail.
- How we test it.
