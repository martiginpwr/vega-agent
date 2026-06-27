import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BookOpen,
  Bot,
  Brain,
  ChevronDown,
  Database,
  Hammer,
  Loader2,
  MessageSquarePlus,
  Mic,
  PanelRight,
  Plus,
  Send,
  Settings,
  SlidersHorizontal,
  Sparkles,
  UserRound,
} from "lucide-react";
import "./styles.css";

const seedMessages = [
  {
    role: "assistant",
    content:
      "Hi, I am Vega. I am running locally through Ollama. Pick a model, ask a question, and we will build the agent layer piece by piece.",
  },
];

function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [health, setHealth] = useState(null);
  const [messages, setMessages] = useState(seedMessages);
  const [draft, setDraft] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadStatus() {
      try {
        const [healthResponse, modelsResponse] = await Promise.all([
          fetch("/api/health"),
          fetch("/api/models"),
        ]);

        const healthData = await healthResponse.json();
        setHealth(healthData);

        if (!modelsResponse.ok) {
          throw new Error("Ollama model list is unavailable.");
        }

        const modelsData = await modelsResponse.json();
        setModels(modelsData.models);
        setSelectedModel(modelsData.default_model || modelsData.models[0]?.name || "");
      } catch (err) {
        setError(err.message);
      }
    }

    loadStatus();
  }, []);

  const contextCount = useMemo(() => messages.length, [messages]);

  async function sendMessage(event) {
    event.preventDefault();
    const content = draft.trim();
    if (!content || !selectedModel || isSending) return;

    const nextMessages = [...messages, { role: "user", content }];
    setMessages(nextMessages);
    setDraft("");
    setIsSending(true);
    setError("");

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: selectedModel,
          messages: nextMessages,
          temperature: 0.7,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "The local model request failed.");
      }

      setMessages((current) => [...current, data.message]);
    } catch (err) {
      setError(err.message);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content:
            "I could not complete that local request. Check that Ollama is running and that the selected model is available.",
        },
      ]);
    } finally {
      setIsSending(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Sparkles size={20} />
          </div>
          <span>Vega Agent</span>
          <button className="icon-button" aria-label="Toggle sidebar">
            <PanelRight size={18} />
          </button>
        </div>

        <button className="new-chat">
          <Plus size={20} />
          New chat
        </button>

        <div className="search-box">
          <MessageSquarePlus size={17} />
          <span>Local sessions coming soon</span>
        </div>

        <nav className="session-list" aria-label="Local sessions">
          <p>Today</p>
          <button className="session active">Initial local chat</button>
          <button className="session">Plan memory layer</button>
          <button className="session">Research agent loops</button>
        </nav>

        <section className="model-picker">
          <div className="section-title">
            <span>Models</span>
            <Settings size={16} />
          </div>
          <label className="select-label" htmlFor="model-select">
            Local Ollama model
          </label>
          <select
            id="model-select"
            value={selectedModel}
            onChange={(event) => setSelectedModel(event.target.value)}
          >
            {models.length === 0 ? (
              <option value="">No models found</option>
            ) : (
              models.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.name}
                </option>
              ))
            )}
          </select>
        </section>

        <div className="connection">
          <span className={health?.ollama_connected ? "dot online" : "dot"} />
          {health?.ollama_connected ? "Ollama connected" : "Ollama offline"}
        </div>
      </aside>

      <section className="chat-panel">
        <header className="chat-header">
          <div>
            <h1>Local conversation</h1>
            <p>{selectedModel || "Select a model"} · private on this device</p>
          </div>
          <div className="header-actions">
            <button className="icon-button" aria-label="Chat options">
              <SlidersHorizontal size={18} />
            </button>
            <button className="icon-button" aria-label="Memory notes">
              <BookOpen size={18} />
            </button>
          </div>
        </header>

        <div className="messages" aria-live="polite">
          {messages.map((message, index) => (
            <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
              <div className="avatar" aria-hidden="true">
                {message.role === "user" ? <UserRound size={18} /> : <Bot size={18} />}
              </div>
              <div className="bubble">
                <div className="message-meta">
                  <strong>{message.role === "user" ? "You" : "Vega"}</strong>
                  <span>Local</span>
                </div>
                <p>{message.content}</p>
              </div>
            </article>
          ))}
          {isSending && (
            <article className="message assistant">
              <div className="avatar" aria-hidden="true">
                <Bot size={18} />
              </div>
              <div className="bubble thinking">
                <Loader2 size={16} />
                Thinking locally...
              </div>
            </article>
          )}
        </div>

        {error && <div className="error-banner">{error}</div>}

        <form className="composer" onSubmit={sendMessage}>
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Ask Vega anything local..."
            rows={3}
          />
          <div className="composer-actions">
            <button className="icon-button" type="button" aria-label="Attach local context">
              <Plus size={20} />
            </button>
            <button className="icon-button" type="button" aria-label="Voice input">
              <Mic size={20} />
            </button>
            <button className="send-button" disabled={!draft.trim() || !selectedModel || isSending}>
              {isSending ? <Loader2 size={18} /> : <Send size={18} />}
              Send
            </button>
          </div>
        </form>
      </section>

      <aside className="inspector">
        <section className="local-status">
          <div className="status-heading">
            <span className={health?.ollama_connected ? "dot online" : "dot"} />
            <strong>Local</strong>
          </div>
          <p>Everything stays on this device.</p>
        </section>

        <InspectorBlock icon={<Brain size={18} />} title="Memory">
          <Metric label="Mode" value="Manual soon" />
          <Metric label="Messages" value={contextCount} />
          <Metric label="Compaction" value="Planned" />
        </InspectorBlock>

        <InspectorBlock icon={<Database size={18} />} title="Retrieval">
          <Metric label="Vector store" value="Planned" />
          <Metric label="Embeddings" value="Local" />
          <Metric label="Top K" value="5" />
        </InspectorBlock>

        <InspectorBlock icon={<Hammer size={18} />} title="Tools">
          <Metric label="File system" value="Designing" />
          <Metric label="Shell" value="Human gate" />
          <Metric label="Skills" value="Future" />
        </InspectorBlock>
      </aside>
    </main>
  );
}

function InspectorBlock({ icon, title, children }) {
  return (
    <section className="inspector-block">
      <header>
        <div>
          {icon}
          <strong>{title}</strong>
        </div>
        <ChevronDown size={16} />
      </header>
      {children}
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
