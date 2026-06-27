import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Bot,
  Brain,
  CheckCircle2,
  ChevronDown,
  CircleAlert,
  FileText,
  Info,
  Loader2,
  Menu,
  MessageCircle,
  MessageSquare,
  MessageSquarePlus,
  PanelLeft,
  Search,
  Send,
  Sparkles,
  Wrench,
  X,
} from "lucide-react";
import "./styles.css";

const seedMessages = [];

const quickActions = [
  { label: "Plan Vega memory", icon: Brain },
  { label: "Explain architecture", icon: FileText },
  { label: "Draft next step", icon: Search },
];

function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [health, setHealth] = useState(null);
  const [messages, setMessages] = useState(seedMessages);
  const [draft, setDraft] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 760);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const messagesEndRef = useRef(null);

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

  useEffect(() => {
    const mediaQuery = window.matchMedia("(max-width: 760px)");
    const syncSidebarWithViewport = () => setSidebarOpen(!mediaQuery.matches);

    syncSidebarWithViewport();
    mediaQuery.addEventListener("change", syncSidebarWithViewport);

    return () => mediaQuery.removeEventListener("change", syncSidebarWithViewport);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isSending]);

  const currentModel = useMemo(
    () => models.find((model) => model.name === selectedModel),
    [models, selectedModel],
  );

  const hasMessages = messages.length > 0;

  function startNewChat() {
    setMessages([]);
    setDraft("");
    setError("");
  }

  function useQuickAction(label) {
    setDraft(label);
  }

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
    <main className={sidebarOpen ? "app-shell sidebar-expanded" : "app-shell sidebar-collapsed"}>
      <Sidebar
        onNewChat={startNewChat}
        onToggle={() => setSidebarOpen((value) => !value)}
        open={sidebarOpen}
      />

      <section className="chat-surface" aria-label="Local chat">
        <header className="topbar">
          <button
            className="icon-button mobile-menu"
            type="button"
            aria-label="Toggle sidebar"
            onClick={() => setSidebarOpen((value) => !value)}
          >
            <Menu size={20} />
          </button>
          <button
            className="model-button"
            type="button"
            onClick={() => setDetailsOpen(true)}
            aria-label="Open local model details"
          >
            <span>{selectedModel || "Select model"}</span>
            <ChevronDown size={16} />
          </button>
          <button
            className="icon-button"
            type="button"
            aria-label="Open agent details"
            onClick={() => setDetailsOpen(true)}
          >
            <Info size={19} />
          </button>
        </header>

        <div className={hasMessages ? "conversation" : "conversation empty"}>
          {hasMessages ? (
            <MessageList messages={messages} isSending={isSending} messagesEndRef={messagesEndRef} />
          ) : (
            <Welcome onQuickAction={useQuickAction} />
          )}
        </div>

        {error ? (
          <div className="error-banner" role="status">
            <CircleAlert size={17} />
            <span>{error}</span>
          </div>
        ) : null}

        <form className="composer" onSubmit={sendMessage}>
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendMessage(event);
              }
            }}
            placeholder="Ask anything"
            rows={1}
          />
          <select
            className="model-select"
            aria-label="Select local model"
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
          <button className="send-button" disabled={!draft.trim() || !selectedModel || isSending} aria-label="Send">
            {isSending ? <Loader2 size={20} /> : <Send size={19} />}
          </button>
        </form>
      </section>

      <AgentDetails
        currentModel={currentModel}
        health={health}
        messageCount={messages.length}
        onClose={() => setDetailsOpen(false)}
        open={detailsOpen}
      />
    </main>
  );
}

function Sidebar({ onNewChat, onToggle, open }) {
  return (
    <aside className="sidebar" aria-label="Conversations">
      <div className="sidebar-brand">
        <button
          className="brand-mark"
          type="button"
          aria-label={open ? "Vega home" : "Expand sidebar"}
          onClick={open ? undefined : onToggle}
        >
          <Sparkles size={18} />
        </button>
        {open ? <strong>Vega Agent</strong> : null}
        <button className="icon-button sidebar-toggle" type="button" aria-label="Toggle sidebar" onClick={onToggle}>
          <PanelLeft size={19} />
        </button>
      </div>

      <nav className="primary-nav" aria-label="Primary">
        <button className="nav-item active" type="button" onClick={onNewChat} title="New chat">
          <MessageSquarePlus size={19} />
          {open ? <span>New chat</span> : null}
        </button>
      </nav>

      {open ? (
        <section className="sidebar-note">
          <p>Conversation history will appear here after SQLite persistence lands.</p>
        </section>
      ) : null}

      <div className="profile-dot" aria-label="Local profile">
        MA
      </div>
    </aside>
  );
}

function Welcome({ onQuickAction }) {
  return (
    <section className="welcome">
      <h1>Hey, Martin. Ready to build locally?</h1>
      <div className="quick-actions">
        {quickActions.map(({ label, icon: Icon }) => (
          <button type="button" key={label} onClick={() => onQuickAction(label)}>
            <Icon size={16} />
            <span>{label}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function MessageList({ messages, isSending, messagesEndRef }) {
  return (
    <div className="message-list" aria-live="polite">
      {messages.map((message, index) => (
        <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
          <div className="message-icon" aria-hidden="true">
            {message.role === "user" ? <MessageCircle size={18} /> : <Bot size={18} />}
          </div>
          <div className="message-content">
            <strong>{message.role === "user" ? "You" : "Vega"}</strong>
            <p>{message.content}</p>
          </div>
        </article>
      ))}
      {isSending ? (
        <article className="message assistant">
          <div className="message-icon" aria-hidden="true">
            <Bot size={18} />
          </div>
          <div className="message-content thinking">
            <Loader2 size={17} />
            <span>Thinking locally...</span>
          </div>
        </article>
      ) : null}
      <div ref={messagesEndRef} />
    </div>
  );
}

function AgentDetails({ currentModel, health, messageCount, onClose, open }) {
  return (
    <div className={open ? "details-layer open" : "details-layer"} aria-hidden={!open}>
      <button className="details-backdrop" type="button" aria-label="Close details" onClick={onClose} />
      <aside className="details-drawer" aria-label="Agent details">
        <header>
          <div>
            <h2>Local agent details</h2>
            <p>Runtime modules stay private on this device.</p>
          </div>
          <button className="icon-button" type="button" aria-label="Close details" onClick={onClose}>
            <X size={19} />
          </button>
        </header>

        <DetailRow
          icon={health?.ollama_connected ? CheckCircle2 : CircleAlert}
          label="Ollama"
          value={health?.ollama_connected ? "Connected" : "Offline"}
        />
        <DetailRow icon={Bot} label="Current model" value={currentModel?.name || "None selected"} />
        <DetailRow icon={MessageSquare} label="Conversation messages" value={String(messageCount)} />
        <DetailRow icon={Brain} label="Memory" value="Milestone 2" />
        <DetailRow icon={Wrench} label="Tools" value="Milestone 5" />
      </aside>
    </div>
  );
}

function DetailRow({ icon: Icon, label, value }) {
  return (
    <div className="detail-row">
      <Icon size={18} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
