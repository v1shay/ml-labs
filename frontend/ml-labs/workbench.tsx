"use client";

import { FormEvent, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import katex from "katex";
import * as THREE from "three";
import type { LabRunError, LabRunResult, SourceResolveResult } from "@/lib/ml-labs/types";

const DEFAULT_KAGGLE_INPUT =
  'path = kagglehub.dataset_download("waddahali/kaggle-competition-graph-dataset")';
const DEMO_DATASET_PATH = "/data/demo-churn.csv";

type ShellPhase =
  | "idle"
  | "searching"
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

export function MlLabsWorkbench() {
  const [input, setInput] = useState("");
  const [phase, setPhase] = useState<ShellPhase>("idle");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "control-ready",
      speaker: "control",
      text: "Control online. Send any instruction and I will start dataset inspection.",
    },
  ]);
  const [sourceResolution, setSourceResolution] = useState<SourceResolveResult | null>(null);
  const [activePrompt, setActivePrompt] = useState("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [runResult, setRunResult] = useState<LabRunResult | null>(null);
  const timers = useRef<number[]>([]);

  useEffect(() => {
    return () => clearTimers(timers.current);
  }, []);

  const profile = useMemo(
    () => (sourceResolution ? buildIngestionProfile(sourceResolution) : null),
    [sourceResolution],
  );
  const modelFamilies = useMemo(
    () => (profile ? buildModelFamilies(profile, runResult) : []),
    [profile, runResult],
  );
  const activeAgents = getActiveAgents(phase);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const prompt = input.trim() || "Start dataset inspection and open the modeling surface.";
    const hasKaggleReference = hasExplicitKaggleReference(prompt);
    const kaggleInput = hasKaggleReference ? prompt : DEFAULT_KAGGLE_INPUT;

    clearTimers(timers.current);
    setInput("");
    setActivePrompt(prompt);
    setSourceResolution(null);
    setRunResult(null);
    setErrorMessage(null);
    setPhase("searching");
    setMessages((current) => [
      ...current,
      { id: `user-${Date.now()}`, speaker: "user", text: prompt },
      {
        id: `control-${Date.now()}`,
        speaker: "control",
        text: hasKaggleReference
          ? "Routing the dataset reference into the Data Panel."
          : "ML-Labs Data Panel is locating an inspection-ready table.",
      },
    ]);

    try {
      const { result: resolved, usedLocalFallback } = await resolveSource(
        kaggleInput,
        !hasKaggleReference,
      );
      setSourceResolution(resolved);
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
          text: `Inspection surface locked on ${resolved.selectedFilePath ?? resolved.sourceLabel}. Active agents are reading the field.`,
        },
      ].filter((message): message is ChatMessage => Boolean(message)));
      void attemptModelRun(resolved, prompt);
      schedulePanelSequence();
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

  async function attemptModelRun(resolved: SourceResolveResult, prompt: string) {
    if (!resolved.sourceToken) {
      return;
    }

    try {
      const formData = new FormData();
      formData.set("sourceToken", resolved.sourceToken);
      formData.set("targetColumn", resolved.targetSuggestions[0]?.column ?? resolved.headers.at(-1) ?? "");
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
    schedule(() => setPhase("testingFirstPass"), 43000, timers.current);
    schedule(() => setPhase("recoding"), 48600, timers.current);
    schedule(() => setPhase("testingPassed"), 54000, timers.current);
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
            placeholder="Ask Control to open a data surface..."
            rows={4}
          />
          <button type="submit" disabled={phase === "searching"}>
            {phase === "searching" ? "Resolving" : "Send"}
          </button>
        </form>
      </aside>

      <section className="agent-workspace">
        <header className="agent-statusbar">
          <div>
            <span>Active Agents</span>
            <strong>{activeAgents.join(" / ")}</strong>
          </div>
          <p>{activePrompt || "No active instruction."}</p>
        </header>

        <div className="active-panel-stage">
          {phase === "idle" ? <IdlePanel /> : null}
          {phase === "searching" ? <DataPanelSearch prompt={activePrompt} /> : null}
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
          {phase === "error" ? <ErrorPanel message={errorMessage ?? "Unknown inspection error."} /> : null}
        </div>
      </section>
    </main>
  );
}

function IdlePanel() {
  return (
    <article className="agent-panel compact-panel">
      <span className="panel-kicker">Workbench</span>
      <h1>Control is waiting on a data instruction.</h1>
      <p>
        The workspace stays empty until an agent opens a surface. No queued panels, no visible
        route map.
      </p>
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
  const visibleArtifacts = useProgressiveItems(artifacts, 520, 80);
  const activeArtifact = visibleArtifacts.at(-1) ?? artifacts[0];
  const visibleCodeLines = useProgressiveItems(activeArtifact.content.split("\n"), 64, 120);

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
        <PanelHeader agent="Diagnostic Graph" title={status === "passed" ? "Re-run curve accepted" : "First curve replay"} />
        <ModelCurve model={selected} />
        <div className="selected-rationale">
          <p>Target anchor: {profile.targetColumn}.</p>
          <p>
            {status === "passed"
              ? "The regenerated curve matches the selected family envelope."
              : "The graph is structurally correct, but the artifact manifest needs one refinement."}
          </p>
        </div>
      </article>
    </div>
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
}: {
  profile: IngestionProfile;
  mode: "unstable" | "spatial";
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
      targets[offset] = mode === "spatial" ? particle.x + (particle.cluster - 1) * 1.2 : particle.x;
      targets[offset + 1] = mode === "spatial" ? particle.y * 0.82 : particle.y;
      targets[offset + 2] = mode === "spatial" ? particle.z + particle.cluster * 0.35 : particle.z;
      color.set(particle.color);
      colors[offset] = color.r;
      colors[offset + 1] = color.g;
      colors[offset + 2] = color.b;
    });

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: mode === "spatial" ? 4.8 : 4.2,
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
        if (mode === "spatial") {
          positions[offset] += (targets[offset] - positions[offset]) * 0.018;
          positions[offset + 1] += (targets[offset + 1] - positions[offset + 1]) * 0.018;
          positions[offset + 2] += (targets[offset + 2] - positions[offset + 2]) * 0.018;
        } else {
          positions[offset] += Math.sin(tick * 4 + index) * 0.0035;
          positions[offset + 1] += Math.cos(tick * 3 + index * 0.4) * 0.0035;
          positions[offset + 2] += Math.sin(tick * 2 + index * 0.7) * 0.0035;
        }
      }
      positionAttribute.needsUpdate = true;
      points.rotation.y += mode === "spatial" ? 0.0018 : 0.0026;
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
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [mode, profile]);

  return (
    <div ref={mountRef} className="particle-canvas">
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

async function resolveSource(kaggleInput: string, preferLocalSource = false): Promise<{
  result: SourceResolveResult;
  usedLocalFallback: boolean;
}> {
  let firstPass: SourceResolveResult;
  let usedLocalFallback = preferLocalSource;
  if (preferLocalSource) {
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
  if (phase === "searching" || phase === "streaming") {
    return ["Control", "Data Panel"];
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
  return ["Control"];
}

function buildIngestionProfile(source: SourceResolveResult): IngestionProfile {
  const rows = source.previewRows;
  const targetColumn = source.targetSuggestions[0]?.column ?? source.headers.at(-1) ?? "target";
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

def build_pipeline():
    numeric = [${profile.numericColumns.map((column) => `"${column.name}"`).join(", ")}]
    categorical = [${profile.categoricalColumns.map((column) => `"${column.name}"`).join(", ")}]
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = ${modelFactory}
    return Pipeline([("preprocess", preprocess), ("model", model)])

def main(source_path: str):
    ARTIFACT_DIR.mkdir(exist_ok=True)
    frame = pd.read_csv(source_path)
    x = frame[FEATURES]
    y = frame[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.22, random_state=42, stratify=y)
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = pipeline.predict(x_test)
    manifest = {
        "target": TARGET,
        "model_family": "${model.name}",
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "report": classification_report(y_test, predictions, output_dict=True),
    }${manifestLine}
    joblib.dump(pipeline, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest

if __name__ == "__main__":
    print(json.dumps(main("inspection-surface.csv"), indent=2))
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
- \`model_card.md\` records family choice and risk notes.
- \`requirements.txt\` captures the ML runtime.

Status: ${mode === "refine" ? "refined after testing agent review" : "initial generation"}
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
