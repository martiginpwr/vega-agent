# Roadmap

## Milestone 0: Local Chat Foundation

Status: started.

- Local React chat UI.
- FastAPI backend.
- Ollama health check.
- Ollama local model listing.
- Per-chat model selection.
- Documentation baseline.

## Milestone 1: Conversation Persistence

- Store sessions in local SQLite.
- Add session list, rename, delete, and resume.
- Keep raw transcripts separate from model context.
- Add export/import for local backups.

## Milestone 2: Memory

- Add explicit "save memory" action.
- Store memories in SQLite with type, source, confidence, and timestamps.
- Add retrieval of relevant memories into chat context.
- Add user controls for edit/delete/disable memory.

## Milestone 3: Local RAG

- Add document ingestion.
- Add local chunking.
- Add local embedding model support.
- Persist vector index locally.
- Show retrieved chunks in the inspector.

## Milestone 4: Context Compaction

- Track approximate context budget per selected model.
- Add manual compaction.
- Add automatic compaction near budget thresholds.
- Store compacted summaries as session artifacts.

## Milestone 5: Tool Runtime

- Define typed tool registry.
- Start with read-only tools.
- Add permission classes and user approval.
- Add structured tool-call traces to the UI.

## Milestone 6: Routing And Small Models

- Add deterministic routing rules.
- Add small local classifier experiments for memory candidacy, intent classification, and tool eligibility.
- Compare model cost, latency, and quality.

## Milestone 7: Skill System

- Store skills as local markdown or JSON artifacts.
- Retrieve relevant skills for new tasks.
- Add post-session skill candidate evaluator.
- Keep human approval required for accepted skills.

## Milestone 8: Evaluation Harness

- Add scripted local eval tasks.
- Measure latency, model choice, retrieval relevance, and answer quality.
- Track regressions for memory, RAG, tools, and compaction.
