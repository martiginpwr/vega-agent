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
  Trash2,
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
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [draft, setDraft] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isLoadingConversation, setIsLoadingConversation] = useState(false);
  const [error, setError] = useState("");
  const [traceOpen, setTraceOpen] = useState(false);
  const [trace, setTrace] = useState(null);
  const [isTraceLoading, setIsTraceLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 760);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [thinkingEnabled, setThinkingEnabled] = useState(false);
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
    loadConversations();
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

  const supportsThinking = currentModel?.capabilities?.includes("thinking") ?? false;
  const hasMessages = messages.length > 0;

  useEffect(() => {
    if (!traceOpen || !trace?.run?.id || !hasActiveTraceProcessing(trace)) return undefined;

    const intervalId = window.setInterval(async () => {
      try {
        const response = await fetch(`/api/traces/${trace.run.id}`);
        const data = await response.json();
        if (response.ok) {
          setTrace(data);
        }
      } catch {
        // Keep the existing trace visible. The next poll can recover if this was transient.
      }
    }, 1500);

    return () => window.clearInterval(intervalId);
  }, [traceOpen, trace]);

  async function loadConversations() {
    try {
      const response = await fetch("/api/conversations");
      if (!response.ok) {
        throw new Error("Conversation history is unavailable.");
      }
      const data = await response.json();
      setConversations(data);
    } catch (err) {
      setError(err.message);
    }
  }

  function startNewChat() {
    setMessages([]);
    setDraft("");
    setError("");
    setActiveConversationId(null);
  }

  function useQuickAction(label) {
    setDraft(label);
  }

  async function openConversation(conversationId) {
    setIsLoadingConversation(true);
    setError("");
    try {
      const response = await fetch(`/api/conversations/${conversationId}`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Could not open that conversation.");
      }

      setActiveConversationId(data.conversation.id);
      setMessages(
        data.messages.map((message) => ({
          id: message.id,
          role: message.role,
          content: message.content,
          metadata: message.metadata || {},
        })),
      );
      if (data.conversation.selected_model) {
        setSelectedModel(data.conversation.selected_model);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoadingConversation(false);
    }
  }

  async function openTrace(runId) {
    if (!runId) return;
    setIsTraceLoading(true);
    setTraceOpen(true);
    setTrace(null);
    setError("");
    try {
      const response = await fetch(`/api/traces/${runId}`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Could not load that trace.");
      }
      setTrace(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsTraceLoading(false);
    }
  }

  async function deleteConversation(conversationId) {
    setError("");
    try {
      const response = await fetch(`/api/conversations/${conversationId}`, { method: "DELETE" });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Could not delete that conversation.");
      }

      if (activeConversationId === conversationId) {
        startNewChat();
      }
      await loadConversations();
    } catch (err) {
      setError(err.message);
    }
  }

  async function sendMessage(event) {
    event.preventDefault();
    const content = draft.trim();
    if (!content || !selectedModel || isSending) return;

    const temporaryUserId = `pending-${Date.now()}`;
    const temporaryAssistantId = `pending-assistant-${Date.now()}`;
    const nextMessages = [...messages, { id: temporaryUserId, role: "user", content, metadata: {} }];
    setMessages(nextMessages);
    setDraft("");
    setIsSending(true);
    setError("");

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: selectedModel,
          conversation_id: activeConversationId,
          messages: nextMessages,
          temperature: 0.7,
          think: supportsThinking ? thinkingEnabled : null,
        }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || "The local model request failed.");
      }
      if (!response.body) {
        throw new Error("The local streaming response is unavailable.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let streamRunId = null;
      let assistantStarted = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;
          const data = JSON.parse(line);
          if (data.event === "error") {
            throw new Error(data.error || "The local model request failed.");
          }
          if (data.event === "start") {
            streamRunId = data.run_id;
            setActiveConversationId(data.conversation_id);
            setMessages((current) => [
              ...current.map((message) =>
                message.id === temporaryUserId
                  ? {
                      ...message,
                      id: data.user_message_id,
                      metadata: { ...message.metadata, run_id: data.run_id },
                    }
                  : message,
              ),
              {
                id: temporaryAssistantId,
                role: "assistant",
                content: "",
                metadata: { run_id: data.run_id },
              },
            ]);
            assistantStarted = true;
          }
          if (data.event === "delta") {
            if (!assistantStarted) {
              setMessages((current) => [
                ...current,
                {
                  id: temporaryAssistantId,
                  role: "assistant",
                  content: "",
                  metadata: streamRunId ? { run_id: streamRunId } : {},
                },
              ]);
              assistantStarted = true;
            }
            setMessages((current) =>
              current.map((message) =>
                message.id === temporaryAssistantId
                  ? { ...message, content: `${message.content}${data.content || ""}` }
                  : message,
              ),
            );
          }
          if (data.event === "done") {
            setActiveConversationId(data.conversation_id);
            setMessages((current) =>
              current.map((message) =>
                message.id === temporaryAssistantId
                  ? {
                      ...message,
                      id: data.assistant_message_id,
                      metadata: { ...message.metadata, run_id: data.run_id },
                    }
                  : message,
              ),
            );
          }
        }
      }
      await loadConversations();
    } catch (err) {
      setError(err.message);
      setMessages((current) => [
        ...current.filter((message) => message.id !== temporaryAssistantId),
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
        activeConversationId={activeConversationId}
        conversations={conversations}
        isLoadingConversation={isLoadingConversation}
        onDeleteConversation={deleteConversation}
        onOpenConversation={openConversation}
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
            <MessageList
              messages={messages}
              isSending={isSending}
              messagesEndRef={messagesEndRef}
              onOpenTrace={openTrace}
            />
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

        <form className={supportsThinking ? "composer has-thinking" : "composer"} onSubmit={sendMessage}>
          {supportsThinking ? (
            <button
              className={thinkingEnabled ? "thinking-toggle active" : "thinking-toggle"}
              type="button"
              aria-label={thinkingEnabled ? "Disable thinking mode" : "Enable thinking mode"}
              aria-pressed={thinkingEnabled}
              onClick={() => setThinkingEnabled((value) => !value)}
            >
              <Brain size={18} />
              <span>{thinkingEnabled ? "Thinking" : "Think"}</span>
            </button>
          ) : null}
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
      <TraceDrawer
        isLoading={isTraceLoading}
        onClose={() => setTraceOpen(false)}
        open={traceOpen}
        trace={trace}
      />
    </main>
  );
}

function renderInlineMarkdown(text) {
  const parts = [];
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*)/g;
  let lastIndex = 0;
  let match;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    const token = match[0];
    if (token.startsWith("`")) {
      parts.push(<code key={`${match.index}-code`}>{token.slice(1, -1)}</code>);
    } else {
      parts.push(<strong key={`${match.index}-strong`}>{token.slice(2, -2)}</strong>);
    }
    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return parts;
}

function MarkdownContent({ content }) {
  const blocks = content.split(/```/);
  return (
    <div className="markdown-content">
      {blocks.map((block, blockIndex) => {
        if (blockIndex % 2 === 1) {
          const lines = block.replace(/^\w+\n/, "");
          return <pre key={`code-${blockIndex}`}><code>{lines.trimEnd()}</code></pre>;
        }

        return block
          .split(/\n{2,}/)
          .filter((section) => section.trim())
          .map((section, sectionIndex) => {
            const key = `text-${blockIndex}-${sectionIndex}`;
            const trimmed = section.trim();
            if (/^#{1,3}\s/.test(trimmed)) {
              return <h3 key={key}>{renderInlineMarkdown(trimmed.replace(/^#{1,3}\s/, ""))}</h3>;
            }
            const listLines = trimmed.split("\n").filter((line) => /^[-*]\s+/.test(line.trim()));
            if (listLines.length && listLines.length === trimmed.split("\n").length) {
              return (
                <ul key={key}>
                  {listLines.map((line, index) => (
                    <li key={`${key}-${index}`}>{renderInlineMarkdown(line.trim().replace(/^[-*]\s+/, ""))}</li>
                  ))}
                </ul>
              );
            }
            return <p key={key}>{renderInlineMarkdown(trimmed)}</p>;
          });
      })}
    </div>
  );
}

function hasActiveTraceProcessing(trace) {
  if (!trace) return false;
  if (trace.run?.status === "failed") return false;
  if (trace.run?.status === "started") return true;
  if (!trace.events?.length) return false;

  const lastStatusByStep = new Map();
  for (const event of trace.events) {
    lastStatusByStep.set(event.step, event.status);
  }

  const hasStartedStep = [...lastStatusByStep.values()].some((status) => status === "started");
  if (hasStartedStep) return true;

  const hasMemoryQueue = trace.events.some((event) => event.step === "memory.queue");
  if (!hasMemoryQueue) return false;

  return !trace.events.some((event) => {
    if (event.step === "memory.write" || event.step === "memory.update" || event.step === "memory.dedupe") {
      return event.status === "completed";
    }
    if (event.step === "memory.error") {
      return event.status === "failed";
    }
    if (event.step === "memory.classifier" && event.message.includes("no durable memory")) {
      return event.status === "completed";
    }
    if (event.step === "memory.grounding" && event.message.includes("rejected")) {
      return event.status === "completed";
    }
    if (event.step === "memory.verifier" && event.message.includes("rejected")) {
      return event.status === "completed";
    }
    return false;
  });
}

function Sidebar({
  activeConversationId,
  conversations,
  isLoadingConversation,
  onDeleteConversation,
  onNewChat,
  onOpenConversation,
  onToggle,
  open,
}) {
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
        <section className="conversation-section">
          <p>Recents</p>
          {conversations.length === 0 ? (
            <span className="empty-history">No saved chats yet.</span>
          ) : (
            conversations.map((conversation) => (
              <div
                className={
                  conversation.id === activeConversationId ? "conversation-row selected" : "conversation-row"
                }
                key={conversation.id}
              >
                <button
                  type="button"
                  onClick={() => onOpenConversation(conversation.id)}
                  disabled={isLoadingConversation}
                  title={conversation.title}
                >
                  <span>{conversation.title}</span>
                </button>
                <button
                  className="delete-chat"
                  type="button"
                  aria-label={`Delete ${conversation.title}`}
                  onClick={() => onDeleteConversation(conversation.id)}
                >
                  <Trash2 size={15} />
                </button>
              </div>
            ))
          )}
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

function MessageList({ messages, isSending, messagesEndRef, onOpenTrace }) {
  const lastMessage = messages[messages.length - 1];
  const showThinking = isSending && !(lastMessage?.role === "assistant" && lastMessage.content);

  return (
    <div className="message-list" aria-live="polite">
      {messages.map((message, index) => (
        <article
          className={`message ${message.role} ${message.metadata?.run_id ? "has-trace" : ""}`}
          key={message.id || `${message.role}-${index}`}
          onClick={() => onOpenTrace(message.metadata?.run_id)}
        >
          <div className="message-icon" aria-hidden="true">
            {message.role === "user" ? <MessageCircle size={18} /> : <Bot size={18} />}
          </div>
          <div className="message-content">
            <strong>{message.role === "user" ? "You" : "Vega"}</strong>
            <MarkdownContent content={message.content} />
          </div>
        </article>
      ))}
      {showThinking ? (
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

function TraceDrawer({ isLoading, onClose, open, trace }) {
  const isProcessing = hasActiveTraceProcessing(trace);

  return (
    <div className={open ? "details-layer open trace-layer" : "details-layer trace-layer"} aria-hidden={!open}>
      <button className="details-backdrop" type="button" aria-label="Close trace" onClick={onClose} />
      <aside className="details-drawer trace-drawer" aria-label="Trace log">
        <header>
          <div>
            <h2>Trace log</h2>
            <p>Local backend steps for this message.</p>
          </div>
          <button className="icon-button" type="button" aria-label="Close trace" onClick={onClose}>
            <X size={19} />
          </button>
        </header>

        {isLoading ? (
          <div className="trace-loading">
            <Loader2 size={17} />
            Loading trace...
          </div>
        ) : trace ? (
          <>
            <div className="trace-summary">
              <span>Run</span>
              <strong className={isProcessing ? "trace-live" : ""}>
                {isProcessing ? (
                  <>
                    <Loader2 size={13} />
                    Processing
                  </>
                ) : (
                  trace.run.status
                )}
              </strong>
            </div>
            <div className="trace-events">
              {trace.events.map((event) => (
                <article className={`trace-event status-${event.status}`} key={event.id}>
                  <div>
                    <strong>{event.step}</strong>
                    <span>
                      {event.status === "started" ? <Loader2 size={12} /> : null}
                      {event.status}
                    </span>
                  </div>
                  <p>{event.message}</p>
                  {Object.keys(event.metadata || {}).length > 0 ? (
                    <pre>{JSON.stringify(event.metadata, null, 2)}</pre>
                  ) : null}
                </article>
              ))}
            </div>
          </>
        ) : (
          <p className="trace-empty">No trace selected.</p>
        )}
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
