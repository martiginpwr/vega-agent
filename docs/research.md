# Research Notes

These notes summarize the first research pass for Vega Agent. The goal is not to copy any one system, but to borrow practical patterns that can run locally on consumer hardware.

## Sources Reviewed

- Claude Code positions the agent as a coding system that reads a codebase, edits files, runs commands, and integrates with development tools. Its Agent SDK describes reusable primitives for file access, command execution, web search, editing, an agent loop, and context management.  
  Sources: <https://code.claude.com/docs/en/overview>, <https://code.claude.com/docs/en/agent-sdk/overview>
- OpenAI Codex CLI is a local terminal coding agent that can read, change, and run code in a selected directory. This reinforces the design pattern of a local process with explicit tool boundaries and a clear working directory.  
  Source: <https://developers.openai.com/codex/cli>
- OpenAI Swarm emphasizes lightweight, testable orchestration with two core primitives: agents and handoffs. The cookbook also frames orchestration as routines plus handoffs.  
  Sources: <https://github.com/openai/swarm>, <https://developers.openai.com/cookbook/examples/orchestrating_agents>
- OpenAI Agents SDK documentation describes two orchestration styles: code-driven orchestration and LLM-driven orchestration, with the option to mix both. Vega should use code-driven defaults for reliability, then allow model choice at well-bounded decision points.  
  Source: <https://openai.github.io/openai-agents-python/multi_agent/>
- LangGraph persistence separates short-term thread memory through checkpointers from long-term memory through stores. This maps cleanly to Vega's future conversation state, resumable runs, and durable user memory.  
  Source: <https://docs.langchain.com/oss/python/langgraph/persistence>
- Ollama exposes local model APIs, including `/api/tags` for local model listing, `/api/chat` for chat, and `/api/embed` for embeddings. Ollama's embedding docs recommend local embedding models such as `embeddinggemma`, `qwen3-embedding`, and `all-minilm`.  
  Sources: <https://github.com/ollama/ollama/blob/main/docs/api.md>, <https://docs.ollama.com/capabilities/embeddings>
- Chroma and similar local vector stores can persist embeddings and metadata for RAG. Vega does not need to choose permanently yet, but the first local RAG layer should support persistent on-disk collections.  
  Source: <https://docs.langchain.com/oss/python/integrations/vectorstores/chroma>
- Sentence Transformers and small embedding models such as `all-MiniLM-L6-v2` or `bge-small-en-v1.5` are realistic for fast local retrieval and classification support on consumer machines.  
  Sources: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>, <https://huggingface.co/BAAI/bge-small-en-v1.5>

## Practical Patterns To Adopt

1. Keep the agent loop explicit.
   The model should not directly execute side effects. It should propose tool calls, the runtime validates them, and risky actions go through policy or user approval.

2. Separate orchestration from model providers.
   Ollama is the first provider, but memory, tools, retrieval, compaction, and routing should depend on an internal interface rather than Ollama-specific payloads.

3. Use code-driven routing first.
   Local models can be weaker at reliable tool planning than frontier cloud models. Deterministic routing rules, typed tool schemas, and small classifiers should handle easy decisions before asking the main chat model.

4. Treat context as a budget.
   Conversation history, retrieved memories, files, and tool outputs should be assembled into a context package. When the budget gets tight, the system should compact older turns into a handoff-oriented summary.

5. Make memory a decision, not a log dump.
   Every conversation is history, but not every sentence is memory. Candidate memories should be classified, deduplicated, scored for usefulness, and stored with provenance.

6. Design skills as local artifacts.
   A future Hermes-style skill creator can analyze successful conversations and propose reusable workflows. These should start as human-reviewable markdown or JSON files before the agent is allowed to use them automatically.

7. Keep tools modular and inspectable.
   Tools should be small local functions with typed inputs, clear permissions, structured outputs, and tests. The user should be able to see what each tool can do.

## Hardware Implications

With an RTX 4070 12 GB VRAM and 16 GB RAM, Vega should avoid loading multiple large models at once. A realistic default path is:

- Main chat/reasoning: current Qwen model through Ollama.
- Embeddings: a dedicated local embedding model through Ollama or Sentence Transformers.
- Routing/classification: rules first, then a small local classifier or compact model.
- Reranking: optional and only after basic retrieval works.
- Background skill extraction: asynchronous and user-approved, not part of the hot chat path.
