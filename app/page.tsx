"use client";

import {
  Bot,
  Braces,
  Database,
  FileText,
  Loader2,
  MessageCircle,
  Send,
  Terminal,
  Upload,
} from "lucide-react";
import {
  ChangeEvent,
  DragEvent,
  FormEvent,
  useMemo,
  useRef,
  useState,
} from "react";

type Role = "user" | "assistant" | "system";
type UploadStatus = "idle" | "reading" | "running" | "complete" | "failed";

type ChatMessage = {
  id: string;
  role: Role;
  content: string;
};

type ChatResponse = {
  message?: ChatMessage;
  error?: string;
};

type DatasetProfile = {
  rows?: number;
  columns?: number;
  targetColumn?: string;
  problemType?: string;
};

type AgentTraceItem = {
  agent?: string;
  status?: string;
  message?: string;
};

type LabRunResponse = {
  runId?: string;
  scenario?: string;
  datasetProfile?: DatasetProfile;
  agentTrace?: AgentTraceItem[];
  leaderboard?: unknown[];
  bestModel?: {
    modelName?: string;
    metricName?: string;
    score?: number;
  };
  criticReport?: unknown;
  visualizations?: unknown[];
  predictionInputSchema?: unknown;
  artifacts?: unknown[];
  finalReportMarkdown?: string;
  error?: string;
  details?: string;
};

type NormalizedRun = {
  runId: string;
  scenario: string;
  problemType: string;
  rows: number | null;
  columns: number | null;
  targetColumn: string;
  bestModel: string;
  metric: string;
  score: number | null;
  visualizations: number;
  artifacts: number;
  schemaFields: number;
  trace: AgentTraceItem[];
};

const starterMessages: ChatMessage[] = [
  {
    id: "system-ready",
    role: "system",
    content: "ML-Labs local agent shell is attached to /api/chat and /api/lab/run.",
  },
  {
    id: "assistant-ready",
    role: "assistant",
    content:
      "Ready for dataset ingestion. Drop a CSV, set the target column, then I will replay the backend trace as it comes online.",
  },
];

const pendingStages = ["reading csv", "parsing rows", "profiling schema", "normalizing columns"];

export default function HomePage() {
  const [messages, setMessages] = useState<ChatMessage[]>(starterMessages);
  const [draft, setDraft] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [intentPrompt, setIntentPrompt] = useState("");
  const [kaggleSlug, setKaggleSlug] = useState("");
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>("idle");
  const [uploadResult, setUploadResult] = useState<NormalizedRun | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [streamLines, setStreamLines] = useState<string[]>([
    "$ waiting for dataset",
    "source: csv upload | kaggle connector pending",
  ]);
  const [activeStageIndex, setActiveStageIndex] = useState(0);
  const [iconFailed, setIconFailed] = useState(false);
  const [wordmarkFailed, setWordmarkFailed] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamTimerRef = useRef<number | null>(null);
  const stageTimerRef = useRef<number | null>(null);

  const visibleMessages = useMemo(
    () => messages.filter((message) => message.role !== "system"),
    [messages],
  );
  const canRunDataset = Boolean(selectedFile && targetColumn.trim() && uploadStatus !== "running");
  const activeStageLabel =
    uploadStatus === "running"
      ? pendingStages[activeStageIndex]
      : uploadStatus === "reading"
        ? "streaming csv"
        : uploadStatus;

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

      appendAssistantMessage(data.message.content);
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

    if (!selectedFile && kaggleSlug.trim()) {
      showKagglePending();
      return;
    }

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
    startStageLoop();
    appendAssistantMessage(`Data Intake Agent: ingesting ${selectedFile.name}`);

    try {
      const response = await fetch("/api/lab/run", {
        method: "POST",
        body: formData,
      });
      const data = (await response.json()) as LabRunResponse;

      if (!response.ok) {
        throw new Error(data.details ?? data.error ?? "Dataset run failed.");
      }

      const normalized = normalizeLabRunResult(data, targetColumn);
      setUploadResult(normalized);
      setUploadStatus("complete");
      stopStageLoop();
      appendTraceMessages(normalized.trace);
      setStreamLines((current) => [
        ...current.slice(-28),
        `runId: ${normalized.runId}`,
        `profile: ${normalized.rows ?? "?"} rows x ${normalized.columns ?? "?"} columns`,
        `bestModel: ${normalized.bestModel}`,
        "status: backend trace complete",
      ]);
    } catch (caughtError) {
      const message =
        caughtError instanceof Error ? caughtError.message : "Unknown dataset error.";
      setUploadError(message);
      setUploadStatus("failed");
      stopStageLoop();
      appendAssistantMessage(`Run blocked: ${message}`);
    }
  }

  function handleChatSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendMessage(draft);
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    void acceptFile(file);
  }

  function handleDrop(event: DragEvent<HTMLButtonElement>) {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0] ?? null;
    void acceptFile(file);
  }

  async function acceptFile(file: File | null) {
    clearStreamTimer();
    setSelectedFile(file);
    setUploadResult(null);
    setUploadError(null);

    if (!file) {
      setUploadStatus("idle");
      setStreamLines(["$ waiting for dataset", "source: csv upload | kaggle connector pending"]);
      return;
    }

    setUploadStatus("reading");
    appendAssistantMessage(`Dataset staged: ${file.name}`);

    const preview = await file.text();
    const rows = preview
      .split(/\r?\n/)
      .filter(Boolean)
      .slice(0, 30)
      .map((line, index) => `${String(index).padStart(3, "0")}  ${line}`);

    animateStream([
      `$ open ${file.name}`,
      `bytes: ${file.size}`,
      "preview:",
      ...rows,
      "$ ready for target column",
    ]);
  }

  function showKagglePending() {
    const slug = kaggleSlug.trim();

    appendAssistantMessage(
      slug
        ? `Kaggle connector pending for ${slug}. CSV upload is the active ingestion path in this frontend pass.`
        : "Kaggle connector pending. Enter owner/dataset or upload a CSV.",
    );
    setStreamLines((current) => [
      ...current.slice(-24),
      `$ kaggle pull ${slug || "owner/dataset"}`,
      "connector: pending backend credentials",
      "status: waiting for csv fallback",
    ]);
  }

  function guideWithAgent() {
    const prompt = selectedFile
      ? `Help me set up ${selectedFile.name} for an ML-Labs dataset run.`
      : "Help me choose and prepare a CSV dataset for ML-Labs Step 1.";

    void sendMessage(prompt);
  }

  function appendAssistantMessage(content: string) {
    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "assistant",
        content,
      },
    ]);
  }

  function appendTraceMessages(trace: AgentTraceItem[]) {
    const items = trace.length
      ? trace
      : [{ agent: "Report Agent", status: "complete", message: "Run completed." }];

    items.slice(0, 12).forEach((item, index) => {
      window.setTimeout(() => {
        appendAssistantMessage(
          `${item.agent ?? "Agent"} [${item.status ?? "complete"}]: ${item.message ?? "stage complete"}`,
        );
      }, index * 360);
    });
  }

  function animateStream(lines: string[]) {
    clearStreamTimer();
    setStreamLines([]);

    let index = 0;
    streamTimerRef.current = window.setInterval(() => {
      setStreamLines((current) => [...current.slice(-34), lines[index]]);
      index += 1;

      if (index >= lines.length) {
        clearStreamTimer();
        setUploadStatus("idle");
      }
    }, 55);
  }

  function startStageLoop() {
    stopStageLoop();
    setActiveStageIndex(0);
    stageTimerRef.current = window.setInterval(() => {
      setActiveStageIndex((current) => (current + 1) % pendingStages.length);
    }, 720);
  }

  function clearStreamTimer() {
    if (streamTimerRef.current !== null) {
      window.clearInterval(streamTimerRef.current);
      streamTimerRef.current = null;
    }
  }

  function stopStageLoop() {
    if (stageTimerRef.current !== null) {
      window.clearInterval(stageTimerRef.current);
      stageTimerRef.current = null;
    }
  }

  return (
    <main className="ide-shell">
      <header className="topbar">
        <div className="brand brand-left">
          {iconFailed ? (
            <span className="brand-fallback">ML</span>
          ) : (
            <img
              src="/brand/ml-labs-icon.svg"
              alt="ML-Labs icon"
              onError={() => setIconFailed(true)}
            />
          )}
        </div>
        <div className="brand brand-right">
          {wordmarkFailed ? (
            <span className="brand-fallback">ML-Labs</span>
          ) : (
            <img
              src="/brand/ml-labs-wordmark.svg"
              alt="ML-Labs"
              onError={() => setWordmarkFailed(true)}
            />
          )}
        </div>
      </header>

      <section className="agent-panel" aria-label="Agent panel">
        <div className="pane-toolbar">
          <span className="toolbar-title">
            <Bot size={14} />
            agent
          </span>
          <span className="toolbar-status">local</span>
        </div>

        <div className="message-list" aria-live="polite">
          {visibleMessages.map((message) => (
            <article className={`message ${message.role}`} key={message.id}>
              <p>{message.content}</p>
            </article>
          ))}

          {isSending ? (
            <article className="message assistant pending">
              <p>thinking...</p>
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
            placeholder="Plan, build, / for commands, @ for context"
            rows={3}
          />
          <button aria-label="Send message" type="submit" disabled={isSending || draft.trim().length === 0}>
            <Send size={15} />
          </button>
        </form>
      </section>

      <section className="workbench-panel" aria-label="Dataset ingestion workbench">
        <div className="pane-toolbar">
          <span className="toolbar-title">
            <Database size={14} />
            dataset ingestion
          </span>
          <span className="toolbar-status">{uploadStatus}</span>
        </div>

        <div className="workbench-grid">
          <form className="ingest-form" onSubmit={runDataset}>
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
              <Upload size={16} />
              <span>{selectedFile ? selectedFile.name : "select or drop csv"}</span>
            </button>

            <label className="field">
              <span>kaggle slug</span>
              <input
                value={kaggleSlug}
                onChange={(event) => setKaggleSlug(event.target.value)}
                placeholder="owner/dataset"
              />
            </label>

            <label className="field">
              <span>target column</span>
              <input
                value={targetColumn}
                onChange={(event) => setTargetColumn(event.target.value)}
                placeholder="charges"
              />
            </label>

            <label className="field intent-field">
              <span>intent prompt</span>
              <textarea
                value={intentPrompt}
                onChange={(event) => setIntentPrompt(event.target.value)}
                placeholder="Create a model to predict the target column."
                rows={3}
              />
            </label>

            {selectedFile ? (
              <div className="file-summary">
                <FileText size={14} />
                <span>{selectedFile.name}</span>
                <small>{formatFileSize(selectedFile.size)}</small>
              </div>
            ) : null}

            <div className="dataset-actions">
              <button className="secondary-action" type="button" onClick={showKagglePending}>
                <Braces size={14} />
                kaggle connector
              </button>
              <button className="secondary-action" type="button" onClick={guideWithAgent}>
                <MessageCircle size={14} />
                ask agent
              </button>
              <button className="primary-action" type="submit" disabled={!canRunDataset}>
                {uploadStatus === "running" ? <Loader2 className="spin" size={14} /> : <Terminal size={14} />}
                run dataset
              </button>
            </div>
          </form>

          <div className="stream-panel">
          <div className="stream-status">
              <span>{activeStageLabel}</span>
              <i />
            </div>
            <pre aria-label="CSV ingestion stream">
              {streamLines.map((line, index) => (
                <code key={`${line}-${index}`}>{line}</code>
              ))}
            </pre>
          </div>
        </div>

        <div className="run-output" aria-live="polite">
          {uploadResult ? (
            <>
              <Metric label="run" value={uploadResult.runId} />
              <Metric label="task" value={uploadResult.problemType} />
              <Metric
                label="shape"
                value={`${uploadResult.rows ?? "?"} x ${uploadResult.columns ?? "?"}`}
              />
              <Metric label="target" value={uploadResult.targetColumn} />
              <Metric label="best" value={uploadResult.bestModel} />
              <Metric label="visuals" value={String(uploadResult.visualizations)} />
              <Metric label="artifacts" value={String(uploadResult.artifacts)} />
              <Metric label="schema" value={String(uploadResult.schemaFields)} />
            </>
          ) : null}

          {uploadStatus === "failed" && uploadError ? (
            <p className="run-error">{uploadError}</p>
          ) : null}

          {!uploadResult && uploadStatus !== "failed" ? (
            <p className="empty-run">awaiting csv stream or kaggle connector</p>
          ) : null}
        </div>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <span className="metric">
      <small>{label}</small>
      <strong>{value}</strong>
    </span>
  );
}

function normalizeLabRunResult(result: LabRunResponse, fallbackTarget: string): NormalizedRun {
  const profile = result.datasetProfile ?? {};
  const schemaFields = countSchemaFields(result.predictionInputSchema);

  return {
    runId: result.runId ?? "local-run",
    scenario: result.scenario ?? profile.problemType ?? "unknown",
    problemType: profile.problemType ?? result.scenario ?? "unknown",
    rows: typeof profile.rows === "number" ? profile.rows : null,
    columns: typeof profile.columns === "number" ? profile.columns : null,
    targetColumn: profile.targetColumn ?? fallbackTarget,
    bestModel: result.bestModel?.modelName ?? "pending",
    metric: result.bestModel?.metricName ?? "metric",
    score: typeof result.bestModel?.score === "number" ? result.bestModel.score : null,
    visualizations: result.visualizations?.length ?? 0,
    artifacts: result.artifacts?.length ?? 0,
    schemaFields,
    trace: result.agentTrace ?? [],
  };
}

function countSchemaFields(schema: unknown) {
  if (!schema) {
    return 0;
  }

  if (Array.isArray(schema)) {
    return schema.length;
  }

  if (typeof schema === "object") {
    if ("fields" in schema && Array.isArray((schema as { fields?: unknown }).fields)) {
      return (schema as { fields: unknown[] }).fields.length;
    }

    return Object.keys(schema).length;
  }

  return 0;
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
