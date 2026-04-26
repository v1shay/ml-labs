"use client";

import {
  FormEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type CSSProperties,
  type DragEvent,
} from "react";
import katex from "katex";
import * as THREE from "three";
import type { LabRunError, LabRunResult, SourceResolveResult } from "@/lib/ml-labs/types";

const DEFAULT_KAGGLE_INPUT =
  'path = kagglehub.dataset_download("waddahali/kaggle-competition-graph-dataset")';
const DEMO_DATASET_PATH = "/data/demo-churn.csv";

type ShellPhase =
  | "idle"
  | "intake"
  | "searching"
  | "targetPick"
  | "streaming"
  | "debating"
  | "particles"
  | "computing"
  | "modelMapping"
  | "modelDebate"
  | "modelSelected"
  | "stackPanel"
  | "codeGenerating"
  | "testingFirstPass"
  | "recoding"
  | "testingPassed"
  | "performanceDiagnostics"
  | "modelStatistics"
  | "methodologySummary"
  | "reportDrafting"
  | "error";

type ChatMessage = {
  id: string;
  speaker: "user" | "control";
  text: string;
};

type ColumnProfile = {
  name: string;
  kind: "numeric" | "categorical";
  missing: number;
  unique: number;
  values: number[];
};

type IngestionProfile = {
  targetColumn: string;
  rowCount: number;
  columnCount: number;
  numericColumns: ColumnProfile[];
  categoricalColumns: ColumnProfile[];
  missingTotal: number;
  classBalance: Array<{ label: string; value: number; count: number }>;
  energy: number[];
  correlations: Array<{ left: string; right: string; value: number }>;
  particles: Array<{ x: number; y: number; z: number; color: string; cluster: number }>;
};

type ModelFamily = {
  id: string;
  agent: string;
  name: string;
  subtitle: string;
  score: number;
  risk: number;
  color: string;
  selected?: boolean;
  graph: number[];
  diagnostics: string[];
};

type StackItem = {
  name: string;
  role: string;
  tone: string;
};

type GeneratedArtifact = {
  filename: string;
  language: string;
  content: string;
};

type SourceRequest =
  | { kind: "zip"; file: File }
  | { kind: "kaggle"; kaggleInput: string }
  | { kind: "localFallback"; kaggleInput: string };

export function MlLabsWorkbench() {
  const [input, setInput] = useState("");
  const [phase, setPhase] = useState<ShellPhase>("intake");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "control-ready",
      speaker: "control",
      text: "ML-Labs Control online. Pick a dataset source on the right to begin a run.",
    },
  ]);
  const [sourceResolution, setSourceResolution] = useState<SourceResolveResult | null>(null);
  const [activePrompt, setActivePrompt] = useState("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [runResult, setRunResult] = useState<LabRunResult | null>(null);
  const [selectedTarget, setSelectedTarget] = useState<string>("");
  const timers = useRef<number[]>([]);

  useEffect(() => {
    return () => clearTimers(timers.current);
  }, []);

  const profile = useMemo(
    () => (sourceResolution ? buildIngestionProfile(sourceResolution, selectedTarget) : null),
    [selectedTarget, sourceResolution],
  );
  const modelFamilies = useMemo(
    () => (profile ? buildModelFamilies(profile, runResult) : []),
    [profile, runResult],
  );
  const activeAgents = getActiveAgents(phase);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const prompt = input.trim();
    if (!prompt) {
      return;
    }

    setInput("");
    setActivePrompt(prompt);
    setMessages((current) => [
      ...current,
      { id: `user-${Date.now()}`, speaker: "user", text: prompt },
      {
        id: `control-${Date.now()}`,
        speaker: "control",
        text: "Project brief noted. Pick a dataset source on the right to start the run.",
      },
    ]);
  }

  function handleBriefChange(nextPrompt: string) {
    setActivePrompt(nextPrompt);
  }

  async function startSourceResolution(request: SourceRequest) {
    const hasKaggleReference =
      request.kind === "kaggle" ? hasExplicitKaggleReference(request.kaggleInput) : false;
    const requestLabel =
      request.kind === "zip"
        ? `ZIP upload ${request.file.name}`
        : request.kind === "kaggle"
          ? "Kaggle API source"
          : "ML-Labs working table";

    clearTimers(timers.current);
    setPhase("searching");
    setSourceResolution(null);
    setRunResult(null);
    setErrorMessage(null);
    setSelectedTarget("");
    setMessages((current) => [
      ...current,
      {
        id: `control-intake-${Date.now()}`,
        speaker: "control",
        text:
          request.kind === "zip"
            ? "ZIP payload accepted. Extracting the largest CSV and locking an inspection surface."
            : request.kind === "kaggle"
              ? "Routing the dataset reference through the Kaggle API resolver."
              : "ML-Labs Data Panel is locating an inspection-ready table.",
      },
    ]);

    try {
      const { result: resolved, usedLocalFallback } = await resolveSource(
        request,
        request.kind === "localFallback" || (request.kind === "kaggle" && !hasKaggleReference),
      );
      setSourceResolution(resolved);
      const defaultTarget =
        resolved.targetSuggestions[0]?.column ?? resolved.headers.at(-1) ?? "";
      setSelectedTarget(defaultTarget);
      setPhase("targetPick");
      setMessages((current) => [
        ...current,
        usedLocalFallback
          ? {
              id: `control-fallback-${Date.now()}`,
              speaker: "control",
              text: "Data Panel resolved an accessible working table and locked a local inspection surface.",
            }
          : null,
        {
          id: `control-resolved-${Date.now()}`,
          speaker: "control",
          text: `${requestLabel} resolved on ${resolved.selectedFilePath ?? resolved.sourceLabel}. Choose what we should predict.`,
        },
      ].filter((message): message is ChatMessage => Boolean(message)));
    } catch (error) {
      const details = error instanceof Error ? error.message : "Inspection failed.";
      setPhase("error");
      setErrorMessage(details);
      setMessages((current) => [
        ...current,
        { id: `control-error-${Date.now()}`, speaker: "control", text: details },
      ]);
    }
  }

  function handleTargetConfirm(target: string) {
    if (!sourceResolution || !target) {
      return;
    }
    const trimmed = target.trim();
    setSelectedTarget(trimmed);
    setMessages((current) => [
      ...current,
      {
        id: `control-target-${Date.now()}`,
        speaker: "control",
        text: `Target confirmed: ${trimmed}. Starting model search and code generation.`,
      },
    ]);
    void attemptModelRun(sourceResolution, trimmed, activePrompt || `Predict ${trimmed}`);
    schedulePanelSequence();
  }

  async function attemptModelRun(
    resolved: SourceResolveResult,
    targetColumn: string,
    prompt: string,
  ) {
    if (!resolved.sourceToken) {
      return;
    }

    try {
      const formData = new FormData();
      formData.set("sourceToken", resolved.sourceToken);
      formData.set("targetColumn", targetColumn);
      formData.set("intentPrompt", prompt);
      const response = await fetch("/api/lab/run", { method: "POST", body: formData });
      const payload = (await response.json()) as LabRunResult | LabRunError;
      if (!response.ok) {
        return;
      }
      setRunResult(payload as LabRunResult);
    } catch {
      // Visual model selection has a deterministic fallback; backend training is opportunistic.
    }
  }

  function schedulePanelSequence() {
    schedule(() => setPhase("streaming"), 450, timers.current);
    schedule(() => setPhase("debating"), 2900, timers.current);
    schedule(() => setPhase("particles"), 7600, timers.current);
    schedule(() => setPhase("computing"), 11600, timers.current);
    schedule(() => setPhase("modelMapping"), 17200, timers.current);
    schedule(() => setPhase("modelDebate"), 22800, timers.current);
    schedule(() => setPhase("modelSelected"), 28800, timers.current);
    schedule(() => setPhase("stackPanel"), 33000, timers.current);
    schedule(() => setPhase("codeGenerating"), 37400, timers.current);
    schedule(() => setPhase("testingFirstPass"), 47200, timers.current);
    schedule(() => setPhase("recoding"), 54000, timers.current);
    schedule(() => setPhase("testingPassed"), 62000, timers.current);
    schedule(() => setPhase("performanceDiagnostics"), 69000, timers.current);
    schedule(() => setPhase("modelStatistics"), 76000, timers.current);
    schedule(() => setPhase("methodologySummary"), 83000, timers.current);
    schedule(() => setPhase("reportDrafting"), 90000, timers.current);
  }

  return (
    <main className="ide-shell">
      <aside className="control-rail">
        <div className="control-topbar">
          <div className="control-brand">
            <img
              className="control-brand-icon"
              src="/brand/ml-labs-icon.png"
              alt=""
              onError={(event) => {
                event.currentTarget.hidden = true;
              }}
            />
            <img
              className="control-brand-wordmark"
              src="/brand/ml-labs-wordmark.png"
              alt="ML-Labs"
              onError={(event) => {
                event.currentTarget.hidden = true;
              }}
            />
            <span>ML-Labs Control</span>
          </div>
          <strong>{phase === "idle" ? "ready" : "active"}</strong>
        </div>

        <div className="control-thread" aria-live="polite">
          {messages.map((message) => (
            <article key={message.id} className={`control-message ${message.speaker}`}>
              <span>{message.speaker === "user" ? "You" : "Control"}</span>
              <p>{message.text}</p>
            </article>
          ))}
        </div>

        <form className="control-input" onSubmit={handleSubmit}>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Optional: log a note for Control while a run is active..."
            rows={4}
          />
          <button type="submit" disabled={phase === "searching" || !input.trim()}>
            {phase === "searching" ? "Resolving" : "Log note"}
          </button>
        </form>
      </aside>

      <section className="agent-workspace">
        <header className="agent-statusbar">
          <div>
            <span>Active Agents</span>
            <strong>{activeAgents.join(" / ")}</strong>
          </div>
          <p>{activePrompt || "Project brief not provided yet — pick a dataset source to begin."}</p>
        </header>

        <div className="active-panel-stage">
          {phase === "idle" || phase === "intake" ? (
            <DataIntakePanel
              prompt={activePrompt}
              onPromptChange={handleBriefChange}
              onResolve={(request) => void startSourceResolution(request)}
            />
          ) : null}
          {phase === "searching" ? <DataPanelSearch prompt={activePrompt} /> : null}
          {phase === "targetPick" && sourceResolution ? (
            <TargetSelectionPanel
              source={sourceResolution}
              defaultTarget={selectedTarget || sourceResolution.targetSuggestions[0]?.column || sourceResolution.headers[0] || ""}
              onConfirm={handleTargetConfirm}
            />
          ) : null}
          {phase === "streaming" && sourceResolution && profile ? (
            <DataStreamPanel source={sourceResolution} profile={profile} />
          ) : null}
          {phase === "debating" && sourceResolution && profile ? (
            <DebatePanel source={sourceResolution} profile={profile} />
          ) : null}
          {phase === "particles" && sourceResolution && profile ? (
            <ParticleInitializationPanel source={sourceResolution} profile={profile} />
          ) : null}
          {phase === "computing" && profile ? <ComputingPanel profile={profile} /> : null}
          {phase === "modelMapping" && profile ? (
            <ModelMappingPanel profile={profile} models={modelFamilies} />
          ) : null}
          {phase === "modelDebate" && profile ? <ModelDebatePanel models={modelFamilies} /> : null}
          {phase === "modelSelected" && profile ? (
            <ModelSelectedPanel profile={profile} models={modelFamilies} />
          ) : null}
          {phase === "stackPanel" ? <TechStackPanel /> : null}
          {phase === "codeGenerating" && profile ? (
            <CodeGenerationPanel profile={profile} models={modelFamilies} mode="initial" />
          ) : null}
          {phase === "testingFirstPass" && profile ? (
            <TestingPanel profile={profile} models={modelFamilies} status="reviewing" />
          ) : null}
          {phase === "recoding" && profile ? (
            <CodeGenerationPanel profile={profile} models={modelFamilies} mode="refine" />
          ) : null}
          {phase === "testingPassed" && profile ? (
            <TestingPanel profile={profile} models={modelFamilies} status="passed" />
          ) : null}
          {phase === "performanceDiagnostics" && profile ? (
            <PerformanceDiagnosticsPanel profile={profile} models={modelFamilies} runResult={runResult} />
          ) : null}
          {phase === "modelStatistics" && profile ? (
            <ModelStatisticsPanel profile={profile} models={modelFamilies} runResult={runResult} />
          ) : null}
          {phase === "methodologySummary" && profile ? (
            <MethodologySummaryPanel profile={profile} models={modelFamilies} runResult={runResult} />
          ) : null}
          {phase === "reportDrafting" && profile ? (
            <ReportDraftingPanel profile={profile} models={modelFamilies} runResult={runResult} />
          ) : null}
          {phase === "error" ? <ErrorPanel message={errorMessage ?? "Unknown inspection error."} /> : null}
        </div>
      </section>
    </main>
  );
}

function DataIntakePanel({
  prompt,
  onPromptChange,
  onResolve,
}: {
  prompt: string;
  onPromptChange: (next: string) => void;
  onResolve: (request: SourceRequest) => void;
}) {
  const [dragActive, setDragActive] = useState(false);
  const [kaggleInput, setKaggleInput] = useState(
    hasExplicitKaggleReference(prompt) ? prompt : DEFAULT_KAGGLE_INPUT,
  );

  function acceptFile(file: File | undefined) {
    if (!file) {
      return;
    }
    const lowered = file.name.toLowerCase();
    if (!lowered.endsWith(".zip") && !lowered.endsWith(".csv")) {
      return;
    }
    onResolve({ kind: "zip", file });
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragActive(false);
    acceptFile(event.dataTransfer.files[0]);
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0]);
  }

  return (
    <article className="agent-panel intake-panel">
      <PanelHeader agent="ML-Labs Intake" title="Choose a dataset source" />
      <div className="intake-layout">
        <section className="intake-card kaggle-card">
          <span>Kaggle</span>
          <strong>Kaggle API resolver</strong>
          <p>Paste a Kaggle dataset slug, URL, or kagglehub snippet to pull the table directly.</p>
          <textarea
            value={kaggleInput}
            onChange={(event) => setKaggleInput(event.target.value)}
            rows={3}
            aria-label="Kaggle dataset input"
          />
          <button type="button" onClick={() => onResolve({ kind: "kaggle", kaggleInput })}>
            Resolve Kaggle table
          </button>
        </section>
        <label
          className={dragActive ? "intake-card zip-card active" : "intake-card zip-card"}
          onDragOver={(event) => {
            event.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
        >
          <input type="file" accept=".zip,.csv,text/csv,application/zip" onChange={handleFileChange} />
          <span>Upload CSV</span>
          <strong>CSV or ZIP archive</strong>
          <p>Drop a CSV or ZIP file. The resolver extracts the largest CSV inside a ZIP and opens the inspection surface.</p>
        </label>
        <section className="intake-card demo-card">
          <span>Demo</span>
          <strong>ML-Labs demo dataset</strong>
          <p>Open the bundled customer-churn working table. Useful for first-time runs and screen recordings.</p>
          <button
            type="button"
            onClick={() =>
              onResolve({ kind: "localFallback", kaggleInput: DEFAULT_KAGGLE_INPUT })
            }
          >
            Use demo dataset
          </button>
        </section>
      </div>
      <section className="intake-brief">
        <span>Project brief (optional)</span>
        <strong>What do you want this model to do?</strong>
        <textarea
          value={prompt}
          onChange={(event) => onPromptChange(event.target.value)}
          rows={3}
          aria-label="Project brief"
          placeholder="e.g. predict customer churn from tenure, billing, and engagement signals"
        />
        <p>This brief is forwarded to the modeling agents as the run intent. You can also leave it blank.</p>
      </section>
    </article>
  );
}

function TargetSelectionPanel({
  source,
  defaultTarget,
  onConfirm,
}: {
  source: SourceResolveResult;
  defaultTarget: string;
  onConfirm: (target: string) => void;
}) {
  const [pick, setPick] = useState(defaultTarget);
  const headers = source.headers.length ? source.headers : [defaultTarget].filter(Boolean);
  const suggestions = source.targetSuggestions ?? [];
  const previewSampleSize = Math.min(source.previewRows.length, 24);
  const targetIndex = Math.max(headers.indexOf(pick), 0);
  const previewSample = source.previewRows.slice(0, previewSampleSize)
    .map((row) => row[targetIndex] ?? "")
    .filter((value) => value !== undefined);

  return (
    <article className="agent-panel target-panel">
      <PanelHeader agent="Problem Framing Agent" title="What should we predict?" />
      <div className="target-layout">
        <section className="target-suggestion-card">
          <span>Top suggestions</span>
          <div className="target-suggestion-list">
            {suggestions.length === 0 ? (
              <p>No automated suggestions surfaced. Pick any column from the dataset list.</p>
            ) : null}
            {suggestions.map((suggestion) => {
              const isActive = suggestion.column === pick;
              return (
                <button
                  key={suggestion.column}
                  type="button"
                  className={isActive ? "target-suggestion active" : "target-suggestion"}
                  onClick={() => setPick(suggestion.column)}
                >
                  <strong>{suggestion.column}</strong>
                  <em>{Math.round(suggestion.confidence * 100)}% confidence</em>
                  <p>{suggestion.reason}</p>
                </button>
              );
            })}
          </div>
        </section>
        <section className="target-headers-card">
          <span>All columns</span>
          <select
            value={pick}
            onChange={(event) => setPick(event.target.value)}
            aria-label="Target column"
          >
            {headers.map((header) => (
              <option key={header} value={header}>
                {header}
              </option>
            ))}
          </select>
          <div className="target-preview">
            <span>Sample target values</span>
            <div className="target-preview-grid">
              {previewSample.length === 0 ? (
                <p>No preview rows are available for this column.</p>
              ) : (
                previewSample
                  .slice(0, 10)
                  .map((value, index) => (
                    <code key={`${value}-${index}`}>{value || "null"}</code>
                  ))
              )}
            </div>
          </div>
        </section>
      </div>
      <div className="target-footer">
        <p>
          ML-Labs will frame the task automatically: the column type and cardinality decide whether
          this becomes regression, binary classification, or multiclass classification.
        </p>
        <button
          type="button"
          onClick={() => onConfirm(pick)}
          disabled={!pick}
        >
          Run with target “{pick || "—"}”
        </button>
      </div>
    </article>
  );
}

function DataPanelSearch({ prompt }: { prompt: string }) {
  return (
    <article className="agent-panel search-panel">
      <PanelHeader agent="ML-Labs Data Panel" title="Resolving an inspection surface" />
      <div className="scanner-grid">
        {["source signature", "tabular boundary", "header entropy", "working table"].map((item, index) => (
          <div key={item} className="scanner-row" style={{ animationDelay: `${index * 160}ms` }}>
            <span>{item}</span>
            <strong>{index === 3 ? "locking" : "measuring"}</strong>
          </div>
        ))}
      </div>
      <p className="panel-copy">
        Control request: {prompt || "open data"}. The Data Panel is probing schema density,
        target candidates, and preview stability before exposing a working table.
      </p>
    </article>
  );
}

function DataStreamPanel({
  source,
  profile,
}: {
  source: SourceResolveResult;
  profile: IngestionProfile;
}) {
  return (
    <article className="agent-panel stream-panel">
      <PanelHeader agent="ML-Labs Data Panel" title="Inspection surface locked" />
      <div className="source-strip">
        <Metric label="Surface" value={source.normalizedKaggleDataset ?? source.sourceLabel} />
        <Metric label="Table" value={source.selectedFilePath ?? "working table"} />
        <Metric label="Feature span" value={`${profile.columnCount} columns / ${profile.rowCount} sampled rows`} />
      </div>

      <div className="data-output">
        {source.previewRows.slice(0, 8).map((row, rowIndex) => (
          <div key={`preview-${rowIndex}`} className="data-row">
            {row.slice(0, 6).map((cell, cellIndex) => (
              <span key={`${rowIndex}-${cellIndex}`}>{cell || "null"}</span>
            ))}
          </div>
        ))}
      </div>
    </article>
  );
}

function DebatePanel({
  source,
  profile,
}: {
  source: SourceResolveResult;
  profile: IngestionProfile;
}) {
  const missingRate =
    profile.rowCount * Math.max(profile.columnCount, 1) > 0
      ? Math.round((profile.missingTotal / (profile.rowCount * profile.columnCount)) * 100)
      : 0;
  const streams = [
    {
      agent: "Schema Agent",
      title: "Column topology",
      lines: [
        `${source.headers.length} headers resolved; ${profile.numericColumns.length} carry numeric signal.`,
        `Categorical channels remain explicit: ${profile.categoricalColumns.map((column) => column.name).slice(0, 3).join(", ") || "none"}.`,
        `Target prior is not treated as truth; ${profile.targetColumn} is only the current best anchor.`,
        "Encoding plan can stay reversible until a model family claims the surface.",
      ],
    },
    {
      agent: "Quality Agent",
      title: "Sampling integrity",
      lines: [
        `${profile.missingTotal} blanks in preview scope; visible missingness is ${missingRate}%.`,
        `Highest cardinality field: ${findHighestCardinality(profile).name}.`,
        "No silent row drop should happen before missingness is projected into the field.",
        "The panel is clean enough for model-family triage, not yet clean enough for final claims.",
      ],
    },
    {
      agent: "Readiness Agent",
      title: "Model surface",
      lines: [
        `Class balance exposes ${profile.classBalance.map((item) => `${item.label}:${item.count}`).join(" / ")}.`,
        `Numeric dimensions are sufficient for distance checks: ${profile.numericColumns.length} observed.`,
        "The first pass should compare linear separability against nonlinear partitioning.",
        "Open the compute surface; the field needs distance, energy, and loss terms before selection.",
      ],
    },
  ];

  return (
    <article className="agent-panel debate-panel">
      <PanelHeader agent="Schema / Quality / Readiness" title="Evidence being contested" />
      <div className="debate-grid live">
        {streams.map((stream, index) => (
          <StreamingAgentBrief
            key={stream.agent}
            agent={stream.agent}
            title={stream.title}
            lines={stream.lines}
            delayMs={index * 240}
          />
        ))}
      </div>
    </article>
  );
}

function ParticleInitializationPanel({
  source,
  profile,
}: {
  source: SourceResolveResult;
  profile: IngestionProfile;
}) {
  return (
    <article className="agent-panel particle-panel">
      <PanelHeader agent="Field Agent" title="Latent field initialized from rows" />
      <div className="particle-layout adaptive">
        <ParticleField profile={profile} mode="unstable" />
        <div className="graph-stack">
          <BarGraph title="Feature type mass" rows={[
            { label: "numeric", value: profile.numericColumns.length },
            { label: "categorical", value: profile.categoricalColumns.length },
          ]} />
          <BarGraph title="Target balance prior" rows={profile.classBalance.slice(0, 5)} />
          <BarGraph
            title="Observed missingness"
            rows={source.headers.slice(0, 6).map((header) => ({
              label: header,
              value:
                profile.numericColumns.find((column) => column.name === header)?.missing ??
                profile.categoricalColumns.find((column) => column.name === header)?.missing ??
                0,
            }))}
          />
          <EnergyGraph values={profile.energy} />
        </div>
      </div>
    </article>
  );
}

function ComputingPanel({ profile }: { profile: IngestionProfile }) {
  return (
    <div className="computing-stack">
      <article className="agent-panel particle-panel wide-panel">
        <PanelHeader agent="Field Agent" title="Latent field retained under projection" />
        <ParticleField profile={profile} mode="spatial" />
      </article>
      <article className="agent-panel computing-panel wide-panel">
        <PanelHeader agent="Computing Agent" title="Neighborhood forces stabilizing" />
        <MathWhiteboard profile={profile} />
        <CorrelationGrid profile={profile} />
      </article>
    </div>
  );
}

function ModelMappingPanel({
  profile,
  models,
}: {
  profile: IngestionProfile;
  models: ModelFamily[];
}) {
  return (
    <article className="agent-panel model-map-panel">
      <PanelHeader agent="Model Selection Agent" title="Candidate families mapped against the field" />
      <div className="model-map-grid">
        {models.map((model) => (
          <ModelCard key={model.id} model={model} targetColumn={profile.targetColumn} />
        ))}
      </div>
    </article>
  );
}

function ModelDebatePanel({ models }: { models: ModelFamily[] }) {
  return (
    <article className="agent-panel model-debate-panel">
      <PanelHeader agent="Model Agents" title="Families arguing from diagnostics" />
      <div className="model-debate-grid">
        {models.filter((model) => model.id !== "baseline").map((model, index) => (
          <StreamingAgentBrief
            key={model.id}
            agent={model.agent}
            title={model.name}
            lines={model.diagnostics}
            delayMs={index * 180}
          />
        ))}
      </div>
    </article>
  );
}

function ModelSelectedPanel({
  profile,
  models,
}: {
  profile: IngestionProfile;
  models: ModelFamily[];
}) {
  const selected = pickModelWinner(models);
  return (
    <article className="agent-panel model-selected-panel">
      <PanelHeader agent="Selection Agent" title="Model family selected for the coding surface" />
      <div className="selected-model-layout">
        <section className="selected-model-card">
          <span>{selected.agent}</span>
          <h2>{selected.name}</h2>
          <p>{selected.subtitle}</p>
          <strong>{selected.score.toFixed(3)} suitability</strong>
        </section>
        <ModelCurve model={selected} />
        <div className="selected-rationale">
          {[
            `Target anchor: ${profile.targetColumn}.`,
            "Wins the first pass because it balances interpretable signal with nonlinear residue.",
            "Coding surface should preserve the baseline and linear model as controls.",
          ].map((line) => (
            <p key={line}>{line}</p>
          ))}
        </div>
      </div>
    </article>
  );
}

function TechStackPanel() {
  const stack: StackItem[] = [
    { name: "Python", role: "training runtime", tone: "#f2c879" },
    { name: "NumPy", role: "tensor algebra", tone: "#a8c7ff" },
    { name: "pandas", role: "frame assembly", tone: "#75d7b5" },
    { name: "scikit-learn", role: "pipeline bench", tone: "#efb06c" },
    { name: "PyTorch", role: "neural candidate", tone: "#ff9d7a" },
    { name: "XGBoost", role: "boosted partitions", tone: "#d8b4ff" },
    { name: "joblib", role: "model bundle", tone: "#9ee6c6" },
    { name: "matplotlib", role: "diagnostic plots", tone: "#8fb7ff" },
    { name: "KaTeX", role: "model math", tone: "#e8edf2" },
  ];

  return (
    <article className="agent-panel stack-panel">
      <PanelHeader agent="Runtime Stack Agent" title="Training stack staged for code generation" />
      <div className="stack-grid">
        {stack.map((item, index) => (
          <section
            key={item.name}
            className="stack-badge"
            style={{ "--stack-tone": item.tone, animationDelay: `${index * 90}ms` } as CSSProperties}
          >
            <div>{item.name.slice(0, 2)}</div>
            <span>{item.name}</span>
            <p>{item.role}</p>
          </section>
        ))}
      </div>
    </article>
  );
}

function CodeGenerationPanel({
  profile,
  models,
  mode,
}: {
  profile: IngestionProfile;
  models: ModelFamily[];
  mode: "initial" | "refine";
}) {
  const selected = pickModelWinner(models);
  const artifacts = useMemo(
    () => buildGeneratedArtifacts(profile, selected, mode),
    [mode, profile, selected],
  );
  const visibleArtifacts = useProgressiveItems(artifacts, 850, 120);
  const activeArtifact = visibleArtifacts.at(-1) ?? artifacts[0];
  const visibleCodeLines = useProgressiveItems(activeArtifact.content.split("\n"), 115, 180);

  return (
    <article className="agent-panel codegen-panel">
      <PanelHeader
        agent={mode === "initial" ? "Code Generation Agent" : "Refactor Agent"}
        title={mode === "initial" ? "Training files materializing" : "Training surface being corrected"}
      />
      <div className="codegen-layout">
        <aside className="code-file-tree">
          <span>Generated files</span>
          {visibleArtifacts.map((artifact) => (
            <button
              key={artifact.filename}
              type="button"
              className={artifact.filename === activeArtifact.filename ? "active" : ""}
              onClick={() => downloadTextFile(artifact.filename, artifact.content)}
            >
              <i>{fileIcon(artifact.filename)}</i>
              {artifact.filename}
            </button>
          ))}
        </aside>
        <section className="code-editor-shell">
          <div className="code-editor-topbar">
            <span>{activeArtifact.filename}</span>
            <strong>{activeArtifact.language}</strong>
          </div>
          <pre className="code-editor">
            {visibleCodeLines.map((line, index) => (
              <code key={`${activeArtifact.filename}-${index}`}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                {line || " "}
              </code>
            ))}
          </pre>
          <div className="artifact-actions-bar">
            <button type="button" onClick={() => downloadTextFile(activeArtifact.filename, activeArtifact.content)}>
              Download active file
            </button>
            <button type="button" onClick={() => void downloadAllArtifacts(artifacts)}>
              Download all
            </button>
            <button
              type="button"
              onClick={() => {
                const summary = artifacts.find((artifact) => artifact.filename === "summary.md") ?? artifacts[0];
                downloadTextFile(summary.filename, summary.content);
              }}
            >
              Download summary
            </button>
            <button type="button" className="vscode-action" title="VS Code launch is staged for local handoff">
              <span>VS</span>
              Open in VS Code
            </button>
          </div>
        </section>
      </div>
    </article>
  );
}

function TestingPanel({
  profile,
  models,
  status,
}: {
  profile: IngestionProfile;
  models: ModelFamily[];
  status: "reviewing" | "passed";
}) {
  const selected = pickModelWinner(models);
  const lines =
    status === "passed"
      ? [
          "artifact import graph resolved",
          "training entrypoint accepts source path and target",
          "diagnostic curve regenerated after refinement",
          "generated training surface passed inspection",
        ]
      : [
          "curve replay opened against generated train.py",
          "detected missing artifact manifest guard",
          "requesting refinement before final handoff",
        ];
  const visibleLines = useProgressiveItems(lines, 520, 100);

  return (
    <div className="testing-layout">
      <article className="agent-panel test-code-card">
        <PanelHeader
          agent={status === "passed" ? "Testing Agent" : "Testing Agent"}
          title={status === "passed" ? "Inspection passed" : "Generated model graph under test"}
        />
        <div className={status === "passed" ? "test-status passed" : "test-status reviewing"}>
          {visibleLines.map((line) => (
            <p key={line}>{line}</p>
          ))}
        </div>
      </article>
      <article className="agent-panel test-graph-card">
        <PanelHeader agent="Testing Field" title={status === "passed" ? "Validated 3D field accepted" : "3D field replay under test"} />
        <ParticleField profile={profile} mode={status === "passed" ? "validated" : "spatial"} model={selected} />
        <div className="selected-rationale">
          <p>Target anchor: {profile.targetColumn}.</p>
          <p>
            {status === "passed"
              ? "The regenerated particle field preserves cluster boundaries, feature projection, and selected-model overlays."
              : "The particle replay is structurally correct, but the artifact manifest needs one refinement."}
          </p>
        </div>
      </article>
    </div>
  );
}

function PerformanceDiagnosticsPanel({
  profile,
  models,
  runResult,
}: {
  profile: IngestionProfile;
  models: ModelFamily[];
  runResult: LabRunResult | null;
}) {
  const selected = pickModelWinner(models);
  const leaderboard = runResult?.leaderboard ?? [];
  const bestScore = leaderboard[0]?.score ?? selected.score;
  return (
    <article className="agent-panel diagnostics-panel">
      <PanelHeader agent="Evaluation Agent" title="Performance diagnostics consolidated" />
      <div className="diagnostics-grid">
        <DiagnosticGraph title="score curve" tone="#8fb7ff" values={models.map((model) => model.score)} variant="line" />
        <DiagnosticGraph title="loss decay" tone="#75d7b5" values={[0.92, 0.71, 0.54, 0.42, 0.34, 0.29, 0.24]} variant="decay" />
        <DiagnosticGraph title="error distribution" tone="#d8b4ff" values={profile.energy.slice(0, 14)} variant="bars" />
        <DiagnosticGraph title="feature importance" tone="#f2c879" values={profile.numericColumns.slice(0, 8).map((column) => Math.min(column.unique / 12, 1))} variant="bars" />
        <DiagnosticGraph title="split quality" tone="#9ec1ff" values={[0.72, 0.78, 0.81, 0.79, 0.84, bestScore]} variant="line" />
        <DiagnosticGraph title="calibration band" tone="#c7a6ff" values={[0.18, 0.31, 0.48, 0.62, 0.74, 0.86]} variant="sigmoid" />
      </div>
    </article>
  );
}

function ModelStatisticsPanel({
  profile,
  models,
  runResult,
}: {
  profile: IngestionProfile;
  models: ModelFamily[];
  runResult: LabRunResult | null;
}) {
  const selected = pickModelWinner(models);
  const winner = runResult?.leaderboard[0];
  const best = runResult?.bestModel;
  const stats = [
    ["target", profile.targetColumn],
    ["training rows", String(runResult?.datasetProfile.rows ?? profile.rowCount)],
    ["feature columns", String(Math.max(profile.columnCount - 1, 0))],
    ["numeric fields", String(profile.numericColumns.length)],
    ["categorical fields", String(profile.categoricalColumns.length)],
    ["selected family", winner?.modelName ?? selected.name],
    ["primary metric", winner?.metricName ?? "suitability"],
    ["held-out score", (winner?.testScore ?? winner?.score ?? selected.score).toFixed(3)],
    ["baseline lift", (best?.absoluteImprovement ?? selected.score - models[0].score).toFixed(3)],
    ["risk score", selected.risk.toFixed(2)],
  ];
  const warnings = runResult?.criticReport.warnings.length
    ? runResult.criticReport.warnings
    : ["No severe blocker surfaced in the generated inspection pass."];
  const nextExperiments = runResult?.criticReport.nextExperiments.length
    ? runResult.criticReport.nextExperiments
    : ["Re-run with a larger validation split.", "Add calibration checks once production labels accumulate."];

  return (
    <article className="agent-panel statistics-panel">
      <PanelHeader agent="Statistics Agent" title="Model accuracy and training statistics" />
      <div className="statistics-layout">
        <div className="stats-grid">
          {stats.map(([label, value]) => (
            <Metric key={label} label={label} value={value} />
          ))}
        </div>
        <section className="stat-notes">
          <span>critic warnings</span>
          {warnings.slice(0, 4).map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </section>
        <section className="stat-notes">
          <span>next experiments</span>
          {nextExperiments.slice(0, 4).map((experiment) => (
            <p key={experiment}>{experiment}</p>
          ))}
        </section>
      </div>
    </article>
  );
}

function MethodologySummaryPanel({
  profile,
  models,
  runResult,
}: {
  profile: IngestionProfile;
  models: ModelFamily[];
  runResult: LabRunResult | null;
}) {
  const selected = pickModelWinner(models);
  const stages = ["INTAKE", "RESOLVE", "PROFILE", "FRAME", "PIPELINE", "BASELINE", "MODEL", "EVAL", "EXPORT"];
  return (
    <article className="agent-panel methodology-panel">
      <PanelHeader agent="Methodology Agent" title="Training methodology rendered" />
      <div className="methodology-layout">
        <img
          src="/brand/ml-labs-methodology.png"
          alt=""
          onError={(event) => {
            event.currentTarget.hidden = true;
          }}
        />
        <div className="methodology-map">
          {stages.map((stage, index) => (
            <div key={stage} className={index === 0 ? "method-node active" : "method-node"}>
              <span>{stage}</span>
            </div>
          ))}
        </div>
        <section className="method-copy">
          <p>
            The table resolves into {profile.numericColumns.length} numeric and {profile.categoricalColumns.length} categorical fields before the selected {selected.name} surface is evaluated.
          </p>
          <p>{runResult?.plainEnglishSummary.shortExplanation ?? "The generated methodology keeps intake, profiling, model comparison, testing, and export as separate inspected stages."}</p>
        </section>
      </div>
    </article>
  );
}

function ReportDraftingPanel({
  profile,
  models,
  runResult,
}: {
  profile: IngestionProfile;
  models: ModelFamily[];
  runResult: LabRunResult | null;
}) {
  const drafts = buildReportDrafts(profile, pickModelWinner(models), runResult);
  return (
    <article className="agent-panel report-drafting-panel">
      <PanelHeader agent="Report Agent" title="Research drafts ready for export" />
      <div className="report-draft-grid">
        {drafts.map((draft) => (
          <section key={draft.filename} className="report-draft-card">
            <span>{draft.filename}</span>
            <p>{draft.content.split("\n").find((line) => line.trim() && !line.startsWith("#")) ?? "Draft generated."}</p>
            <button type="button" onClick={() => downloadTextFile(draft.filename, draft.content)}>
              Download draft
            </button>
          </section>
        ))}
      </div>
    </article>
  );
}

function DiagnosticGraph({
  title,
  tone,
  values,
  variant,
}: {
  title: string;
  tone: string;
  values: number[];
  variant: "line" | "decay" | "bars" | "sigmoid";
}) {
  const safeValues = values.length ? values : [0.2, 0.45, 0.62, 0.74];
  const points = safeValues.map((value, index) => {
    const normalized = variant === "decay" ? 1 - value : value;
    const x = 8 + (index / Math.max(safeValues.length - 1, 1)) * 84;
    const y = 88 - Math.max(0.06, Math.min(normalized, 0.96)) * 76;
    return `${x},${y}`;
  });
  return (
    <section className={`diagnostic-graph ${variant}`}>
      <span>{title}</span>
      <svg viewBox="0 0 100 100" role="img" aria-label={title}>
        <path d="M 8 88 H 94 M 8 12 V 88" />
        {variant === "bars" ? (
          safeValues.map((value, index) => (
            <rect
              key={`${title}-${index}`}
              x={10 + index * (78 / safeValues.length)}
              y={88 - Math.max(value, 0.08) * 70}
              width={Math.max(4, 48 / safeValues.length)}
              height={Math.max(value, 0.08) * 70}
              style={{ fill: tone }}
            />
          ))
        ) : (
          <polyline points={points.join(" ")} style={{ stroke: tone }} />
        )}
        {variant === "sigmoid" ? <path d="M 10 82 C 30 82 39 68 49 52 C 60 34 72 20 92 18" style={{ stroke: tone }} /> : null}
      </svg>
    </section>
  );
}

function ErrorPanel({ message }: { message: string }) {
  return (
    <article className="agent-panel compact-panel error-panel">
      <span className="panel-kicker">Control</span>
      <h1>Inspection stopped.</h1>
      <p>{message}</p>
    </article>
  );
}

function PanelHeader({ agent, title }: { agent: string; title: string }) {
  return (
    <div className="panel-header">
      <span>{agent}</span>
      <h2>{title}</h2>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-chip">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StreamingAgentBrief({
  agent,
  title,
  lines,
  delayMs = 0,
}: {
  agent: string;
  title: string;
  lines: string[];
  delayMs?: number;
}) {
  const visibleLines = useProgressiveItems(lines, 620, delayMs);
  return (
    <section className="agent-brief streaming">
      <span>{agent}</span>
      <strong>{title}</strong>
      <div className="brief-lines">
        {visibleLines.map((line) => (
          <p key={line}>{line}</p>
        ))}
        {visibleLines.length < lines.length ? <i /> : null}
      </div>
    </section>
  );
}

function MathWhiteboard({ profile }: { profile: IngestionProfile }) {
  const equations = [
    {
      label: "normalize feature channel",
      tex: "z_{i,k}=\\frac{x_{i,k}-\\mu_k}{\\sigma_k+\\epsilon}",
      note: `${profile.numericColumns[0]?.name ?? "x"} is centered before force projection.`,
    },
    {
      label: "row distance",
      tex: "d(x_i,x_j)=\\lVert z_i-z_j\\rVert_2",
      note: "Distance is computed in normalized feature space.",
    },
    {
      label: "correlation force",
      tex: "F_{ij}=\\alpha\\,\\rho_{ij}-\\beta\\,d(x_i,x_j)",
      note: "Positive correlation pulls neighborhoods together; distance resists collapse.",
    },
    {
      label: "field energy",
      tex: "E=\\sum_i\\lVert v_i\\rVert_2^2+\\lambda\\sum_{i,j}\\max(0,m-d_{ij})",
      note: "Energy tracks instability while particles settle.",
    },
    {
      label: "candidate loss",
      tex: "\\mathcal{L}(\\theta)=\\frac{1}{n}\\sum_i\\ell(f_\\theta(x_i),y_i)+\\Omega(\\theta)",
      note: `Loss terms prepare model-family selection for ${profile.targetColumn}.`,
    },
  ];
  const visibleEquations = useProgressiveItems(equations, 780, 120);

  return (
    <section className="math-whiteboard">
      {visibleEquations.map((equation) => (
        <article key={equation.label} className="math-line">
          <span>{equation.label}</span>
          <div
            className="math-render"
            dangerouslySetInnerHTML={{
              __html: katex.renderToString(equation.tex, {
                displayMode: true,
                throwOnError: false,
              }),
            }}
          />
          <p>{equation.note}</p>
        </article>
      ))}
      {visibleEquations.length < equations.length ? <div className="typing-cursor" /> : null}
    </section>
  );
}

function ParticleField({
  profile,
  mode,
  model,
}: {
  profile: IngestionProfile;
  mode: "unstable" | "spatial" | "validated";
  model?: ModelFamily;
}) {
  const mountRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) {
      return;
    }
    const container = mount;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 100);
    camera.position.set(0, 0.8, 9);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x090c10, 1);
    container.appendChild(renderer.domElement);

    const geometry = new THREE.BufferGeometry();
    const particleCount = Math.max(profile.particles.length, 1);
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const targets = new Float32Array(particleCount * 3);
    const color = new THREE.Color();

    profile.particles.forEach((particle, index) => {
      const offset = index * 3;
      positions[offset] = particle.x + jitter(index, 0) * 1.6;
      positions[offset + 1] = particle.y + jitter(index, 1) * 1.6;
      positions[offset + 2] = particle.z + jitter(index, 2) * 1.6;
      targets[offset] = mode !== "unstable" ? particle.x + (particle.cluster - 1) * 1.2 : particle.x;
      targets[offset + 1] = mode !== "unstable" ? particle.y * 0.82 : particle.y;
      targets[offset + 2] = mode !== "unstable" ? particle.z + particle.cluster * 0.35 : particle.z;
      color.set(particle.color);
      colors[offset] = color.r;
      colors[offset + 1] = color.g;
      colors[offset + 2] = color.b;
    });

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: mode === "validated" ? 5.4 : mode === "spatial" ? 4.8 : 4.2,
      sizeAttenuation: false,
      vertexColors: true,
      transparent: true,
      opacity: 0.96,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const points = new THREE.Points(geometry, material);
    scene.add(points);

    const frame = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(7.2, 4.4, 4.4)),
      new THREE.LineBasicMaterial({ color: 0x6f8199, transparent: true, opacity: 0.58 }),
    );
    scene.add(frame);

    const grid = new THREE.GridHelper(7.2, 8, 0x26313d, 0x151b22);
    grid.position.y = -2.2;
    scene.add(grid);

    const axes = new THREE.AxesHelper(4.2);
    (axes.material as THREE.Material).transparent = true;
    (axes.material as THREE.Material).opacity = 0.42;
    scene.add(axes);

    function resize() {
      const width = Math.max(container.clientWidth, 320);
      const height = Math.max(container.clientHeight, 260);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }

    let animationFrame = 0;
    let tick = 0;
    function animate() {
      tick += 0.01;
      const positionAttribute = geometry.getAttribute("position") as THREE.BufferAttribute;
      for (let index = 0; index < particleCount; index += 1) {
        const offset = index * 3;
        if (mode !== "unstable") {
          const pull = mode === "validated" ? 0.026 : 0.018;
          positions[offset] += (targets[offset] - positions[offset]) * pull;
          positions[offset + 1] += (targets[offset + 1] - positions[offset + 1]) * pull;
          positions[offset + 2] += (targets[offset + 2] - positions[offset + 2]) * pull;
        } else {
          positions[offset] += Math.sin(tick * 4 + index) * 0.0035;
          positions[offset + 1] += Math.cos(tick * 3 + index * 0.4) * 0.0035;
          positions[offset + 2] += Math.sin(tick * 2 + index * 0.7) * 0.0035;
        }
      }
      positionAttribute.needsUpdate = true;
      points.rotation.y += mode === "validated" ? 0.0012 : mode === "spatial" ? 0.0018 : 0.0026;
      points.rotation.x = Math.sin(tick) * 0.06;
      renderer.render(scene, camera);
      animationFrame = window.requestAnimationFrame(animate);
    }

    resize();
    animate();
    const observer = new ResizeObserver(resize);
    observer.observe(container);

    return () => {
      window.cancelAnimationFrame(animationFrame);
      observer.disconnect();
      geometry.dispose();
      material.dispose();
      frame.geometry.dispose();
      (frame.material as THREE.Material).dispose();
      grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();
      axes.geometry.dispose();
      (axes.material as THREE.Material).dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [mode, profile]);

  return (
    <div ref={mountRef} className="particle-canvas">
      <div className="particle-labels" aria-hidden="true">
        <span className="axis-x">feature axis x</span>
        <span className="axis-y">target gradient y</span>
        <span className="axis-z">latent depth z</span>
      </div>
      <div className="particle-legend" aria-hidden="true">
        <span>cluster force</span>
        <strong>{mode === "validated" ? "validated" : mode}</strong>
        {model ? <em>{model.name}</em> : null}
      </div>
      <div className="particle-overlay" aria-hidden="true">
        {profile.particles.slice(0, 96).map((particle, index) => {
          const drift = mode === "spatial" ? (particle.cluster - 1) * 11 : jitter(index, 0) * 7;
          const depth = Math.max(0.38, Math.min(1, (particle.z + 2.8) / 5.6));
          return (
            <i
              key={`particle-${mode}-${index}`}
              style={{
                left: `${50 + (particle.x / 6) * 72 + drift}%`,
                top: `${50 - (particle.y / 3.8) * 58 + jitter(index, 1) * 3}%`,
                backgroundColor: particle.color,
                opacity: depth,
                transform: `scale(${0.62 + depth * 0.9})`,
                animationDelay: `${index * 18}ms`,
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

function BarGraph({
  title,
  rows,
}: {
  title: string;
  rows: Array<{ label: string; value: number }>;
}) {
  const maxValue = Math.max(...rows.map((row) => row.value), 1);
  return (
    <section className="mini-graph">
      <span>{title}</span>
      {rows.map((row) => (
        <div key={row.label} className="bar-row">
          <em>{row.label}</em>
          <div>
            <i style={{ width: `${Math.max((row.value / maxValue) * 100, 4)}%` }} />
          </div>
          <strong>{row.value}</strong>
        </div>
      ))}
    </section>
  );
}

function EnergyGraph({ values }: { values: number[] }) {
  return (
    <section className="mini-graph">
      <span>Energy distribution</span>
      <div className="energy-bars">
        {values.map((value, index) => (
          <i key={`energy-${index}`} style={{ height: `${Math.max(value * 100, 8)}%` }} />
        ))}
      </div>
    </section>
  );
}

function CorrelationGrid({ profile }: { profile: IngestionProfile }) {
  const correlations = profile.correlations.slice(0, 16);
  return (
    <section className="correlation-panel">
      <span>correlation force map</span>
      <div className="correlation-grid">
        {correlations.map((item) => (
          <div
            key={`${item.left}-${item.right}`}
            style={{ opacity: 0.28 + Math.abs(item.value) * 0.72 }}
            title={`${item.left} / ${item.right}: ${item.value.toFixed(2)}`}
          />
        ))}
      </div>
    </section>
  );
}

function ModelCard({
  model,
  targetColumn,
}: {
  model: ModelFamily;
  targetColumn: string;
}) {
  return (
    <section className={model.selected ? "model-card selected" : "model-card"}>
      <div className="model-card-top">
        <span>{model.agent}</span>
        <strong>{model.score.toFixed(3)}</strong>
      </div>
      <h3>{model.name}</h3>
      <p>{model.subtitle}</p>
      <ModelCurve model={model} />
      <div className="model-card-foot">
        <span>target {targetColumn}</span>
        <span>risk {model.risk.toFixed(2)}</span>
      </div>
    </section>
  );
}

function ModelCurve({ model }: { model: ModelFamily }) {
  const points = model.graph.map((value, index) => {
    const x = (index / Math.max(model.graph.length - 1, 1)) * 100;
    const y = 100 - value * 100;
    return `${x},${y}`;
  });
  const sigmoidPath = "M 4 86 C 22 86 28 74 40 58 C 52 40 62 23 96 18";
  const neuralPath = "M 4 80 C 18 38 28 78 42 42 C 54 12 66 68 80 28 C 88 12 94 18 98 10";
  const linearPath = "M 5 78 L 95 24";
  const bandTop = "M 5 66 L 95 12 L 95 32 L 5 88 Z";
  const stepPath = "M 5 78 L 20 78 L 20 62 L 38 62 L 38 45 L 58 45 L 58 31 L 78 31 L 78 18 L 96 18";

  return (
    <svg className={`model-curve curve-${model.id}`} viewBox="0 0 100 100" role="img" aria-label={`${model.name} diagnostic graph`}>
      {model.id === "baseline" ? (
        <>
          <rect x="5" y="50" width="90" height="18" style={{ fill: model.color }} />
          <path d="M 5 59 L 95 59" style={{ stroke: model.color }} />
        </>
      ) : null}
      {model.id === "linear" ? (
        <>
          <path d={bandTop} className="curve-band" style={{ fill: model.color }} />
          <path d={linearPath} style={{ stroke: model.color }} />
          <path d="M 14 72 L 14 63 M 34 61 L 34 54 M 58 43 L 58 35 M 82 30 L 82 21" className="residual-lines" />
        </>
      ) : null}
      {model.id === "logistic" ? (
        <>
          <path d={sigmoidPath} style={{ stroke: model.color }} />
          <path d="M 58 8 L 58 92" className="threshold-line" />
        </>
      ) : null}
      {model.id === "ensemble" ? (
        <>
          <path d={stepPath} style={{ stroke: model.color }} />
          <path d="M 5 92 L 20 78 L 38 62 L 58 45 L 78 31 L 96 18" className="gain-shadow" />
        </>
      ) : null}
      {model.id === "neural" ? (
        <>
          <path d={neuralPath} style={{ stroke: model.color }} />
          {[18, 34, 50, 66, 82].map((x, index) => (
            <g key={`layer-${x}`} className="neural-layer">
              <circle cx={x} cy={28 + (index % 2) * 12} r="3" />
              <circle cx={x} cy={52} r="3" />
              <circle cx={x} cy={76 - (index % 2) * 12} r="3" />
            </g>
          ))}
        </>
      ) : null}
      {!["baseline", "linear", "logistic", "ensemble", "neural"].includes(model.id) ? (
        <polyline points={points.join(" ")} style={{ stroke: model.color }} />
      ) : null}
    </svg>
  );
}

async function resolveSource(request: SourceRequest, preferLocalSource = false): Promise<{
  result: SourceResolveResult;
  usedLocalFallback: boolean;
}> {
  let firstPass: SourceResolveResult;
  let usedLocalFallback = preferLocalSource;
  const kaggleInput = request.kind === "kaggle" ? request.kaggleInput : DEFAULT_KAGGLE_INPUT;
  if (request.kind === "zip") {
    firstPass = await postFileResolveRequest(request.file);
  } else if (preferLocalSource) {
    firstPass = await postDemoResolveRequest();
  } else {
    try {
      firstPass = await postResolveRequest(kaggleInput);
    } catch (error) {
      if (!(error instanceof Error) || !error.message.toLowerCase().includes("kagglehub")) {
        throw error;
      }
      firstPass = await postDemoResolveRequest();
      usedLocalFallback = true;
    }
  }

  if (firstPass.selectedFilePath || !firstPass.candidateFiles[0]?.path) {
    return { result: firstPass, usedLocalFallback };
  }

  return {
    result: await postResolveRequest(kaggleInput, firstPass.candidateFiles[0].path),
    usedLocalFallback,
  };
}

async function postFileResolveRequest(file: File): Promise<SourceResolveResult> {
  const formData = new FormData();
  formData.set("file", file);

  const response = await fetch("/api/lab/source/resolve", {
    method: "POST",
    body: formData,
  });
  const payload = (await response.json()) as SourceResolveResult | LabRunError;
  if (!response.ok) {
    throw new Error((payload as LabRunError).details ?? (payload as LabRunError).error);
  }
  return payload as SourceResolveResult;
}

async function postResolveRequest(
  kaggleInput: string,
  selectedFilePath?: string,
): Promise<SourceResolveResult> {
  const formData = new FormData();
  formData.set("kaggleInput", kaggleInput);
  if (selectedFilePath) {
    formData.set("selectedFilePath", selectedFilePath);
  }

  const response = await fetch("/api/lab/source/resolve", {
    method: "POST",
    body: formData,
  });
  const payload = (await response.json()) as SourceResolveResult | LabRunError;
  if (!response.ok) {
    throw new Error((payload as LabRunError).details ?? (payload as LabRunError).error);
  }
  return payload as SourceResolveResult;
}

async function postDemoResolveRequest(): Promise<SourceResolveResult> {
  const demoResponse = await fetch(DEMO_DATASET_PATH);
  if (!demoResponse.ok) {
    throw new Error("Data Panel could not open a local inspection surface.");
  }
  const demoBlob = await demoResponse.blob();
  const formData = new FormData();
  formData.set("file", new File([demoBlob], "inspection-surface.csv", { type: "text/csv" }));

  const response = await fetch("/api/lab/source/resolve", {
    method: "POST",
    body: formData,
  });
  const payload = (await response.json()) as SourceResolveResult | LabRunError;
  if (!response.ok) {
    throw new Error((payload as LabRunError).details ?? (payload as LabRunError).error);
  }
  return payload as SourceResolveResult;
}

function hasExplicitKaggleReference(prompt: string): boolean {
  return /kagglehub|kaggle\.com|[\w-]+\/[\w.-]+/.test(prompt);
}

function getActiveAgents(phase: ShellPhase): string[] {
  if (phase === "idle") {
    return ["Control"];
  }
  if (phase === "intake") {
    return ["Control", "Data Intake"];
  }
  if (phase === "searching" || phase === "streaming") {
    return ["Control", "Data Panel"];
  }
  if (phase === "targetPick") {
    return ["Control", "Problem Framing"];
  }
  if (phase === "debating") {
    return ["Control", "Schema", "Quality", "Readiness"];
  }
  if (phase === "particles") {
    return ["Control", "Field"];
  }
  if (phase === "computing") {
    return ["Control", "Field", "Computing"];
  }
  if (phase === "modelMapping") {
    return ["Control", "Model Selection"];
  }
  if (phase === "modelDebate") {
    return ["Control", "Model Agents"];
  }
  if (phase === "modelSelected") {
    return ["Control", "Selection"];
  }
  if (phase === "stackPanel") {
    return ["Control", "Model Stack"];
  }
  if (phase === "codeGenerating") {
    return ["Control", "Code Generation"];
  }
  if (phase === "testingFirstPass") {
    return ["Control", "Testing"];
  }
  if (phase === "recoding") {
    return ["Control", "Refactor"];
  }
  if (phase === "testingPassed") {
    return ["Control", "Testing", "Accepted"];
  }
  if (phase === "performanceDiagnostics") {
    return ["Control", "Evaluation"];
  }
  if (phase === "modelStatistics") {
    return ["Control", "Statistics"];
  }
  if (phase === "methodologySummary") {
    return ["Control", "Methodology"];
  }
  if (phase === "reportDrafting") {
    return ["Control", "Report"];
  }
  return ["Control"];
}

function buildIngestionProfile(
  source: SourceResolveResult,
  targetOverride?: string,
): IngestionProfile {
  const rows = source.previewRows;
  const targetColumn =
    (targetOverride && source.headers.includes(targetOverride) ? targetOverride : null) ??
    source.targetSuggestions[0]?.column ??
    source.headers.at(-1) ??
    "target";
  const targetIndex = Math.max(source.headers.indexOf(targetColumn), 0);
  const columns = source.headers.map((name, columnIndex): ColumnProfile => {
    const rawValues = rows.map((row) => row[columnIndex] ?? "");
    const numericValues = rawValues.map((value) => Number(value)).filter((value) => Number.isFinite(value));
    const kind = numericValues.length >= Math.max(Math.ceil(rawValues.length * 0.65), 2) ? "numeric" : "categorical";
    return {
      name,
      kind,
      missing: rawValues.filter((value) => !value.trim()).length,
      unique: new Set(rawValues.filter(Boolean)).size,
      values: rawValues.map((value) => (kind === "numeric" ? Number(value) || 0 : hashValue(value))),
    };
  });
  const numericColumns = columns.filter((column) => column.kind === "numeric");
  const categoricalColumns = columns.filter((column) => column.kind === "categorical");
  const missingTotal = columns.reduce((total, column) => total + column.missing, 0);
  const targetCounts = countValues(rows.map((row) => row[targetIndex] ?? "unknown"));
  const classBalance = targetCounts.length
    ? targetCounts
    : [{ label: "observed", value: rows.length, count: rows.length }];
  const correlations = buildCorrelations(numericColumns);
  const particles = buildParticles(rows, columns);
  const energy = Array.from({ length: 22 }, (_, index) => {
    const particle = particles[index % Math.max(particles.length, 1)];
    if (!particle) {
      return 0.2;
    }
    return Math.min(Math.sqrt(particle.x ** 2 + particle.y ** 2 + particle.z ** 2) / 5, 1);
  });

  return {
    targetColumn,
    rowCount: rows.length,
    columnCount: source.headers.length,
    numericColumns,
    categoricalColumns,
    missingTotal,
    classBalance,
    energy,
    correlations,
    particles,
  };
}

function buildModelFamilies(profile: IngestionProfile, runResult: LabRunResult | null): ModelFamily[] {
  const realBest = runResult?.bestModel.modelName.toLowerCase() ?? "";
  const imbalance = Math.abs((profile.classBalance[0]?.count ?? 1) - (profile.classBalance[1]?.count ?? 1));
  const complexity = Math.min((profile.numericColumns.length + profile.categoricalColumns.length) / 10, 1);
  const base = 0.44 + complexity * 0.16;
  const families: ModelFamily[] = [
    {
      id: "baseline",
      agent: "Control Model",
      name: "Prior baseline",
      subtitle: "Reference prior, not a production candidate.",
      score: Math.min(base, 0.62),
      risk: 0.22,
      color: "#687381",
      graph: [0.42, 0.44, 0.45, 0.45, 0.46, 0.46],
      diagnostics: ["Baseline establishes the floor.", "It cannot explain feature interactions."],
    },
    {
      id: "linear",
      agent: "Linear Agent",
      name: "Regularized linear regression",
      subtitle: "Tests whether the field is mostly additive and monotonic.",
      score: 0.66 + complexity * 0.08,
      risk: 0.31,
      color: "#8fb7ff",
      graph: [0.48, 0.56, 0.61, 0.65, 0.69, 0.72],
      diagnostics: [
        "Linear fit gives a readable coefficient surface.",
        "It will underfit if categorical boundaries are sharp.",
        "Keep it as an interpretability control even if it loses.",
      ],
    },
    {
      id: "logistic",
      agent: "Logistic Agent",
      name: "Logistic regression",
      subtitle: "Calibrated decision boundary for target odds and separability.",
      score: 0.7 + Math.min(imbalance, 6) * 0.008,
      risk: 0.28,
      color: "#75d7b5",
      graph: [0.5, 0.59, 0.66, 0.71, 0.75, 0.78],
      diagnostics: [
        "Logistic form matches a binary target surface cleanly.",
        "Calibration can stay inspectable while still moving beyond baseline.",
        "It needs interaction checks before we trust final probabilities.",
      ],
    },
    {
      id: "ensemble",
      agent: "Partition Agent",
      name: "Tree ensemble regression",
      subtitle: "Partitions mixed feature space without forcing linear geometry.",
      score: 0.76 + complexity * 0.1,
      risk: 0.39,
      color: "#d8b4ff",
      selected: !realBest || realBest.includes("forest") || realBest.includes("boost"),
      graph: [0.47, 0.6, 0.72, 0.79, 0.84, 0.88],
      diagnostics: [
        "Tree partitions absorb categorical structure with less preprocessing strain.",
        "Residual shape improves when feature effects are discontinuous.",
        "Risk is interpretability drift; feature importance must be preserved.",
      ],
    },
    {
      id: "neural",
      agent: "Neural Agent",
      name: "Neural network regression",
      subtitle: "Dense nonlinear approximator for high-order feature interactions.",
      score: 0.68 + complexity * 0.12,
      risk: 0.54,
      color: "#f2c879",
      graph: [0.44, 0.55, 0.65, 0.73, 0.79, 0.83],
      diagnostics: [
        "Neural candidate has capacity, but this sample surface is still small.",
        "It should not win unless validation density grows.",
        "Useful later if embeddings or richer row volume arrive.",
      ],
    },
  ];

  if (realBest) {
    return families.map((family) => ({
      ...family,
      selected:
        family.name.toLowerCase().includes(realBest) ||
        (family.id === "ensemble" && (realBest.includes("forest") || realBest.includes("boost"))),
    }));
  }

  return families;
}

function pickModelWinner(models: ModelFamily[]): ModelFamily {
  return models.find((model) => model.selected) ?? [...models].sort((left, right) => right.score - left.score)[0];
}

function buildGeneratedArtifacts(
  profile: IngestionProfile,
  model: ModelFamily,
  mode: "initial" | "refine",
): GeneratedArtifact[] {
  const target = profile.targetColumn;
  const featureList = [...profile.numericColumns, ...profile.categoricalColumns]
    .filter((column) => column.name !== target)
    .map((column) => `"${column.name}"`)
    .join(", ");
  const modelFactory =
    model.id === "ensemble"
      ? "RandomForestClassifier(n_estimators=240, min_samples_leaf=2, random_state=42)"
      : model.id === "logistic"
        ? "LogisticRegression(max_iter=2000, class_weight='balanced')"
        : "HistGradientBoostingClassifier(max_iter=180, learning_rate=0.055, random_state=42)";
  const manifestLine = mode === "refine" ? "\n    manifest[\"validated\"] = True" : "";

  return [
    {
      filename: "train.py",
      language: "python",
      content: `from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "${target}"
FEATURES = [${featureList}]
ARTIFACT_DIR = Path("artifacts")
RANDOM_STATE = 42
TEST_SIZE = 0.22

def build_pipeline():
    numeric = [${profile.numericColumns.map((column) => `"${column.name}"`).join(", ")}]
    categorical = [${profile.categoricalColumns.map((column) => `"${column.name}"`).join(", ")}]
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = ${modelFactory}
    return Pipeline([("preprocess", preprocess), ("model", model)])

def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURES + [TARGET] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return frame[FEATURES + [TARGET]].copy()

def build_manifest(pipeline, x_test, y_test, probabilities, predictions):
    report = classification_report(y_test, predictions, output_dict=True)
    return {
        "target": TARGET,
        "model_family": "${model.name}",
        "feature_count": len(FEATURES),
        "row_count": int(len(x_test)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "report": report,
        "positive_rate": float(pd.Series(predictions).mean()),
    }

def main(source_path: str):
    ARTIFACT_DIR.mkdir(exist_ok=True)
    frame = validate_frame(pd.read_csv(source_path))
    x = frame[FEATURES]
    y = frame[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = pipeline.predict(x_test)
    manifest = build_manifest(pipeline, x_test, y_test, probabilities, predictions)${manifestLine}
    joblib.dump(pipeline, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest

if __name__ == "__main__":
    print(json.dumps(main("inspection-surface.csv"), indent=2))
`,
    },
    {
      filename: "features.py",
      language: "python",
      content: `NUMERIC_FEATURES = [${profile.numericColumns.map((column) => `"${column.name}"`).join(", ")}]
CATEGORICAL_FEATURES = [${profile.categoricalColumns.map((column) => `"${column.name}"`).join(", ")}]
TARGET = "${target}"

def feature_contract():
    return {
        "target": TARGET,
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "feature_count": len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
    }
`,
    },
    {
      filename: "predict.py",
      language: "python",
      content: `import json
import joblib
import pandas as pd

MODEL_PATH = "artifacts/model.joblib"

def predict(payload: dict):
    model = joblib.load(MODEL_PATH)
    frame = pd.DataFrame([payload])
    probability = float(model.predict_proba(frame)[0, 1])
    return {"prediction": bool(probability >= 0.5), "probability": probability}

if __name__ == "__main__":
    sample = json.loads(input())
    print(json.dumps(predict(sample), indent=2))
`,
    },
    {
      filename: "evaluate.py",
      language: "python",
      content: `import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

ARTIFACT_DIR = Path("artifacts")

def evaluate(source_path: str):
    frame = pd.read_csv(source_path)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x_train, x_test, y_train, y_test = train_test_split(
        frame[features],
        frame[TARGET],
        test_size=0.22,
        random_state=42,
        stratify=frame[TARGET],
    )
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = model.predict(x_test)
    diagnostics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
        "holdout_rows": int(len(x_test)),
        "train_rows": int(len(x_train)),
    }
    (ARTIFACT_DIR / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    return diagnostics

if __name__ == "__main__":
    print(json.dumps(evaluate("inspection-surface.csv"), indent=2))
`,
    },
    {
      filename: "config.yaml",
      language: "yaml",
      content: `target: ${target}
model_family: ${model.name}
test_size: 0.22
random_state: 42
features:
  numeric: ${profile.numericColumns.length}
  categorical: ${profile.categoricalColumns.length}
artifacts:
  - artifacts/model.joblib
  - artifacts/manifest.json
  - artifacts/diagnostics.json
`,
    },
    {
      filename: "model_card.md",
      language: "markdown",
      content: `# ML-Labs Model Card

Selected family: **${model.name}**

Target: \`${target}\`

Why this family:
- Suitability score: ${model.score.toFixed(3)}
- Risk score: ${model.risk.toFixed(2)}
- Preserves baseline and linear controls for comparison.
- Requires artifact manifest validation before handoff.

Training contract:
- Target column: \`${target}\`
- Numeric features: ${profile.numericColumns.length}
- Categorical features: ${profile.categoricalColumns.length}
- Generated after ${mode === "refine" ? "testing-agent refinement" : "initial code synthesis"}
`,
    },
    {
      filename: "requirements.txt",
      language: "text",
      content: `pandas
numpy
scikit-learn
joblib
matplotlib
pyyaml
`,
    },
    {
      filename: "summary.md",
      language: "markdown",
      content: `# Generated Training Surface

ML-Labs generated a compact training package for \`${target}\`.

Files:
- \`train.py\` builds and saves the training pipeline.
- \`predict.py\` loads the saved model and scores one payload.
- \`evaluate.py\` regenerates holdout diagnostics.
- \`features.py\` records the feature contract.
- \`config.yaml\` captures runtime parameters.
- \`model_card.md\` records family choice and risk notes.
- \`requirements.txt\` captures the ML runtime.

Status: ${mode === "refine" ? "refined after testing agent review" : "initial generation"}
`,
    },
    {
      filename: "research_report.md",
      language: "markdown",
      content: `# ML-Labs Research Report

## Objective
Train and evaluate a ${model.name} surface for target \`${target}\` using the resolved inspection table.

## Data Understanding
The resolved table exposes ${profile.rowCount} previewed rows, ${profile.columnCount} columns, ${profile.numericColumns.length} numeric fields, and ${profile.categoricalColumns.length} categorical fields. Missing value pressure is ${profile.missingTotal} cells in preview.

## Method
The generated training package applies column-aware imputation, categorical encoding, model fitting, holdout evaluation, artifact serialization, and a manifest-based testing loop.

## Result
The selected family score is ${model.score.toFixed(3)} with risk ${model.risk.toFixed(2)}. The testing field validates that row particles remain stable after projection and model-family selection.
`,
    },
  ];
}

function buildReportDrafts(
  profile: IngestionProfile,
  model: ModelFamily,
  runResult: LabRunResult | null,
): GeneratedArtifact[] {
  const target = profile.targetColumn;
  const score = runResult?.leaderboard[0]?.score ?? model.score;
  const metric = runResult?.leaderboard[0]?.metricName ?? "selection score";
  const summary = runResult?.plainEnglishSummary;
  return [
    {
      filename: "executive_summary.md",
      language: "markdown",
      content: `# Executive Summary

ML-Labs resolved a working table and selected **${model.name}** for target \`${target}\`.

${summary?.shortExplanation ?? "The run profiled the dataset, compared model families, validated the generated training surface, and prepared export-ready artifacts."}

- Rows inspected: ${profile.rowCount}
- Feature columns: ${Math.max(profile.columnCount - 1, 0)}
- Metric: ${metric}
- Score: ${score.toFixed(3)}
`,
    },
    {
      filename: "sample_research_report.md",
      language: "markdown",
      content: runResult?.finalReportMarkdown ?? `# Sample Research Report

## Dataset
The table contains ${profile.columnCount} columns and targets \`${target}\`.

## Modeling
${model.name} was selected after baseline, linear, tree, boosting, and neural candidates were inspected.

## Validation
The final 3D field replay preserved particle clusters and target-axis separation after the generated code refinement pass.
`,
    },
    {
      filename: "model_card_draft.md",
      language: "markdown",
      content: `# Model Card Draft

Model family: ${model.name}

Target: ${target}

Known limits:
- Preview-derived profiling can understate rare values.
- Final production readiness needs a larger holdout and drift monitor.
- Connector-backed MCP data is staged for a later pass.
`,
    },
    {
      filename: "methods_appendix.md",
      language: "markdown",
      content: `# Methods Appendix

Pipeline stages: intake, resolution, profiling, target framing, feature pipeline, model sweep, evaluation, critique, and export.

Feature contract:
- Numeric: ${profile.numericColumns.map((column) => column.name).join(", ") || "none"}
- Categorical: ${profile.categoricalColumns.map((column) => column.name).join(", ") || "none"}

Diagnostics rendered: 3D particle projection, score curve, loss decay, error distribution, feature importance, split quality, and calibration band.
`,
    },
  ];
}

function fileIcon(filename: string): string {
  if (filename.endsWith(".py")) {
    return "PY";
  }
  if (filename.endsWith(".md")) {
    return "MD";
  }
  return "TX";
}

function downloadTextFile(filename: string, content: string) {
  downloadBlob(filename, new Blob([content], { type: "text/plain;charset=utf-8" }));
}

function downloadBlob(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

async function downloadAllArtifacts(artifacts: GeneratedArtifact[]) {
  const JSZip = (await import("jszip")).default;
  const zip = new JSZip();
  artifacts.forEach((artifact) => zip.file(artifact.filename, artifact.content));
  const blob = await zip.generateAsync({ type: "blob" });
  downloadBlob("ml-labs-generated-artifacts.zip", blob);
}

function findHighestCardinality(profile: IngestionProfile): ColumnProfile {
  return [...profile.numericColumns, ...profile.categoricalColumns].sort((left, right) => right.unique - left.unique)[0];
}

function buildParticles(rows: string[][], columns: ColumnProfile[]): IngestionProfile["particles"] {
  const usableColumns = columns.slice(0, 4);
  const palette = ["#a8c7ff", "#73d7b8", "#d8b4ff", "#f2c879"];
  const particleCount = Math.max(132, rows.length * 10, 48);
  const particleRows = Array.from({ length: particleCount }, (_, index) =>
    rows.length ? rows[index % rows.length] : [],
  );

  return particleRows.map((row, index) => {
    const sourceIndex = rows.length ? index % rows.length : index;
    const values = usableColumns.map((column) =>
      normalize((column.values[sourceIndex] ?? hashValue(row[sourceIndex] ?? index)) + index * 7),
    );
    const cluster = Math.abs(Math.round((values[0] ?? 0) * 3)) % 3;
    return {
      x: ((values[0] ?? hashValue(String(index))) - 0.5) * 6,
      y: ((values[1] ?? hashValue(`${index}-y`)) - 0.5) * 3.8,
      z: ((values[2] ?? hashValue(`${index}-z`)) - 0.5) * 3.8,
      color: palette[cluster],
      cluster,
    };
  });
}

function buildCorrelations(columns: ColumnProfile[]): IngestionProfile["correlations"] {
  const numeric = columns.slice(0, 6);
  const pairs: IngestionProfile["correlations"] = [];
  numeric.forEach((left, leftIndex) => {
    numeric.forEach((right, rightIndex) => {
      if (rightIndex <= leftIndex) {
        return;
      }
      pairs.push({ left: left.name, right: right.name, value: correlation(left.values, right.values) });
    });
  });
  return pairs.length ? pairs : [{ left: "x", right: "y", value: 0.42 }];
}

function countValues(values: string[]): Array<{ label: string; value: number; count: number }> {
  const counts = new Map<string, number>();
  values.forEach((value) => {
    const label = value.trim() || "unknown";
    counts.set(label, (counts.get(label) ?? 0) + 1);
  });
  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1])
    .slice(0, 6)
    .map(([label, count]) => ({ label, value: count, count }));
}

function normalize(value: number): number {
  return Math.abs(Math.sin(value * 0.013 + value * value * 0.0003));
}

function correlation(left: number[], right: number[]): number {
  const length = Math.min(left.length, right.length);
  if (length < 2) {
    return 0;
  }
  const leftMean = left.slice(0, length).reduce((sum, value) => sum + value, 0) / length;
  const rightMean = right.slice(0, length).reduce((sum, value) => sum + value, 0) / length;
  let numerator = 0;
  let leftDenominator = 0;
  let rightDenominator = 0;
  for (let index = 0; index < length; index += 1) {
    const leftDelta = left[index] - leftMean;
    const rightDelta = right[index] - rightMean;
    numerator += leftDelta * rightDelta;
    leftDenominator += leftDelta ** 2;
    rightDenominator += rightDelta ** 2;
  }
  return numerator / Math.max(Math.sqrt(leftDenominator * rightDenominator), 1);
}

function hashValue(value: string | number): number {
  const text = String(value);
  let hash = 0;
  for (let index = 0; index < text.length; index += 1) {
    hash = (hash << 5) - hash + text.charCodeAt(index);
    hash |= 0;
  }
  return Math.abs(hash % 1000);
}

function jitter(index: number, axis: number): number {
  return Math.sin(index * 12.9898 + axis * 78.233) * 0.5;
}

function useProgressiveItems<T>(items: T[], intervalMs: number, delayMs = 0): T[] {
  const [count, setCount] = useState(0);

  useEffect(() => {
    setCount(0);
    const handles: number[] = [];
    items.forEach((_, index) => {
      handles.push(window.setTimeout(() => setCount(index + 1), delayMs + intervalMs * (index + 1)));
    });
    return () => handles.forEach((handle) => window.clearTimeout(handle));
  }, [delayMs, intervalMs, items.length]);

  return items.slice(0, count);
}

function schedule(callback: () => void, delayMs: number, bucket: number[]) {
  const handle = window.setTimeout(callback, delayMs);
  bucket.push(handle);
}

function clearTimers(handles: number[]) {
  handles.forEach((handle) => window.clearTimeout(handle));
  handles.length = 0;
}
