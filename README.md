# Vega Agent

Vega Agent is a local-first personal AI agent project. The first milestone is intentionally simple: a web chat UI talks to local Ollama models through a local FastAPI backend. The architecture is documented so the project can grow into memory, retrieval, tool use, routing, context compaction, and reusable skills without changing the privacy model.

## Local-Only Promise

- No cloud model calls.
- Ollama is the only model provider in the initial implementation.
- The backend binds to your machine and stores future state under local project-controlled directories.
- Embeddings, vector search, memory, classifiers, and tools are planned as local components.

## Current Features

- ChatGPT-like web interface.
- Lists locally installed Ollama models.
- Selects the active model per conversation.
- Sends chat messages to Ollama's local `/api/chat` endpoint.
- Shows local-only system status and planned agent modules.
- Includes architecture, research, and learning-path documentation.

## Quick Start

Prerequisites:

- Python 3.11+
- Node.js 20+
- Ollama installed and running
- At least one local Ollama chat model, for example your Qwen model

Install backend dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Install frontend dependencies:

```powershell
cd frontend
npm install
```

Run the backend:

```powershell
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Run the frontend in another terminal:

```powershell
cd frontend
npm run dev
```

Open the Vite URL, usually `http://localhost:5173`.

## Configuration

Copy `.env.example` to `.env` if you want to override defaults.

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
VEGA_DEFAULT_MODEL=
```

If `VEGA_DEFAULT_MODEL` is empty, the app uses the first model returned by Ollama.

## Documentation

- [Architecture](docs/architecture.md)
- [Project Plan](docs/project-plan.md)
- [Research Notes](docs/research.md)
- [Roadmap](docs/roadmap.md)
- [Learning Path](docs/learning-path.md)

## Hardware Target

The design targets a consumer local setup:

- RTX 4070 12 GB VRAM
- 16 GB RAM now, possible 32 GB later
- 16-core CPU

This means Vega should prefer cheap local components for small jobs. Large chat models should not be used for every routing, memory, or classification task when smaller local embedding or classifier models can do the job.
