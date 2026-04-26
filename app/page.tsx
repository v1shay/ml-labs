"use client";

import {
  Bot,
  Database,
  FileText,
  Loader2,
  MessageCircle,
  Send,
  Terminal,
  Upload,
} from "lucide-react";
import { ChangeEvent, DragEvent, FormEvent, useMemo, useRef, useState } from "react";

type Role = "user" | "assistant" | "system";

type ChatMessage = {
  id: string;
  role: Role;
  content: string;
};

type ChatResponse = {
  message?: ChatMessage;
  error?: string;
};

type LabRunResponse = {
  runId?: string;
  datasetProfile?: {
    rows: number;
    columns: number;
    targetColumn: string;
    problemType: string;
  };
  bestModel?: {
    modelName: string;
    metricName: string;
    score: number;
  };
  error?: string;
  details?: string;
};

type UploadStatus = "idle" | "running" | "complete" | "failed";

const starterMessages: ChatMessage[] = [
  {
    id: "system-ready",
    role: "system",
    content: "Coding agent shell is connected to /api/chat.",
  },
  {
    id: "assistant-ready",
    role: "assistant",
    content:
      "Step 1 is dataset input. Upload a CSV, name the target column, or ask me to guide the setup.",
  },
];

export default function HomePage() {
  const [messages, setMessages] = useState<ChatMessage[]>(starterMessages);
  const [draft, setDraft] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [intentPrompt, setIntentPrompt] = useState("");
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>("idle");
  const [uploadResult, setUploadResult] = useState<LabRunResponse | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const visibleMessages = useMemo(
    () => messages.filter((message) => message.role !== "system"),
    [messages],
  );
  const canRunDataset = Boolean(selectedFile && targetColumn.trim() && uploadStatus !== "running");

  async function sendMessage(content: string) {
    const trimmed = content.trim();

    if (!trimmed || isSending) {
      return;
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
    };

    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setDraft("");
    setIsSending(true);
    setChatError(null);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: nextMessages.map(({ role, content }) => ({ role, content })),
        }),
      });

      const data = (await response.json()) as ChatResponse;

      if (!response.ok || !data.message) {
        throw new Error(data.error ?? "The chat backend did not return a message.");
      }

      setMessages((current) => [...current, data.message as ChatMessage]);
    } catch (caughtError) {
      const message =
        caughtError instanceof Error ? caughtError.message : "Unknown chat error.";
      setChatError(message);
    } finally {
      setIsSending(false);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }

  async function runDataset(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!selectedFile || !targetColumn.trim()) {
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("targetColumn", targetColumn.trim());

    if (intentPrompt.trim()) {
      formData.append("intentPrompt", intentPrompt.trim());
    }

    setUploadStatus("running");
    setUploadError(null);
    setUploadResult(null);

    try {
      const response = await fetch("/api/lab/run", {
        method: "POST",
        body: formData,
      });
      const data = (await response.json()) as LabRunResponse;

      if (!response.ok) {
        throw new Error(data.details ?? data.error ?? "Dataset run failed.");
      }

      setUploadResult(data);
      setUploadStatus("complete");
    } catch (caughtError) {
      const message =
        caughtError instanceof Error ? caughtError.message : "Unknown dataset error.";
      setUploadError(message);
      setUploadStatus("failed");
    }
  }

  function handleChatSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendMessage(draft);
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setUploadResult(null);
    setUploadError(null);
    setUploadStatus("idle");
  }

  function handleDrop(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0] ?? null;

    if (file) {
      setSelectedFile(file);
      setUploadResult(null);
      setUploadError(null);
      setUploadStatus("idle");
    }
  }

  function guideWithAgent() {
    const prompt = selectedFile
      ? `Help me set up ${selectedFile.name} for an ML-Labs dataset run.`
      : "Help me choose and prepare a CSV dataset for ML-Labs Step 1.";

    void sendMessage(prompt);
  }

  return (
    <main className="app-shell">
      <img className="floating-logo" src="/brand/ml-labs-icon.svg" alt="ML-Labs icon" />

      <section className="agent-panel" aria-label="Agent panel">
        <div className="panel-label">
          <Bot size={18} />
          <span>agent</span>
        </div>

        <div className="message-list" aria-live="polite">
          {visibleMessages.map((message) => (
            <article className={`message ${message.role}`} key={message.id}>
              <p>{message.content}</p>
            </article>
          ))}

          {isSending ? (
            <article className="message assistant pending">
              <p>Thinking...</p>
            </article>
          ) : null}
        </div>

        {chatError ? <p className="error-banner">{chatError}</p> : null}

        <form className="composer" onSubmit={handleChatSubmit}>
          <textarea
            ref={inputRef}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder="Chat to the agent..."
            rows={3}
          />
          <button aria-label="Send message" type="submit" disabled={isSending || draft.trim().length === 0}>
            <Send size={18} />
          </button>
        </form>
      </section>

      <section className="output-panel" aria-label="Dataset input panel">
        <div className="dataset-shell">
          <header className="dataset-header">
            <div>
              <span className="step-kicker">step 01</span>
              <h1>Dataset Input</h1>
            </div>
            <span className="status-pill">
              <Terminal size={15} />
              {uploadStatus}
            </span>
          </header>

          <form className="dataset-form" onSubmit={runDataset}>
            <input
              ref={fileInputRef}
              className="file-input"
              type="file"
              accept=".csv,text/csv"
              onChange={handleFileChange}
              aria-hidden="true"
              hidden
              tabIndex={-1}
            />

            <button
              className="dropzone"
              type="button"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(event) => event.preventDefault()}
              onDrop={handleDrop}
            >
              <span className="drop-icon">
                <Upload size={26} />
              </span>
              <span className="drop-copy">
                <strong>{selectedFile ? selectedFile.name : "Upload dataset"}</strong>
                <small>CSV only for this first demo state</small>
              </span>
            </button>

            {selectedFile ? (
              <div className="file-summary">
                <FileText size={17} />
                <span>{selectedFile.name}</span>
                <small>{formatFileSize(selectedFile.size)}</small>
              </div>
            ) : null}

            <label className="field">
              <span>
                <Database size={15} />
                target column
              </span>
              <input
                value={targetColumn}
                onChange={(event) => setTargetColumn(event.target.value)}
                placeholder="charges"
              />
            </label>

            <label className="field">
              <span>
                <MessageCircle size={15} />
                intent prompt
              </span>
              <textarea
                value={intentPrompt}
                onChange={(event) => setIntentPrompt(event.target.value)}
                placeholder="Create a model to predict the target column."
                rows={4}
              />
            </label>

            <div className="dataset-actions">
              <button className="secondary-action" type="button" onClick={guideWithAgent}>
                <MessageCircle size={16} />
                chat with agent
              </button>
              <button className="primary-action" type="submit" disabled={!canRunDataset}>
                {uploadStatus === "running" ? <Loader2 className="spin" size={16} /> : <Upload size={16} />}
                run dataset
              </button>
            </div>
          </form>

          <div className="run-output" aria-live="polite">
            {uploadStatus === "complete" && uploadResult ? (
              <>
                <span>run initialized</span>
                <p>
                  {uploadResult.datasetProfile?.rows ?? 0} rows,{" "}
                  {uploadResult.datasetProfile?.columns ?? 0} columns, target{" "}
                  {uploadResult.datasetProfile?.targetColumn ?? targetColumn}
                </p>
                {uploadResult.bestModel ? (
                  <small>
                    {uploadResult.bestModel.modelName} · {uploadResult.bestModel.metricName}{" "}
                    {uploadResult.bestModel.score.toFixed(3)}
                  </small>
                ) : null}
              </>
            ) : null}

            {uploadStatus === "failed" && uploadError ? (
              <>
                <span>run blocked</span>
                <p>{uploadError}</p>
              </>
            ) : null}

            {uploadStatus === "idle" ? (
              <>
                <span>awaiting dataset</span>
                <p>Learning begins as particles enter the system.</p>
              </>
            ) : null}
          </div>
        </div>
      </section>

      <img className="wordmark-logo" src="/brand/ml-labs-wordmark.svg" alt="ML-Labs" />
    </main>
  );
}

function formatFileSize(bytes: number) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
