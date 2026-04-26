"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { EvidenceWorkspace } from "@/frontend/ml-labs/components/evidence-workspace";
import { TopologyMatrix } from "@/frontend/ml-labs/components/topology-matrix";
import {
  artifactBundleName,
  buildCandidateOptions,
  buildPredictionSeed,
  buildStageIntel,
  buildSummaryCards,
  coercePredictionPayload,
  downloadableArtifacts,
  getStageStatusMap,
  humanizeTaskSubtype,
  normalizeResolveMessages,
  normalizeRunMessages,
  pickSuggestedTarget,
  STAGES,
  summaryToText,
  type ShellMessage,
  type ShellPhase,
  type SourceMode,
  visibleVisualizations,
} from "@/frontend/ml-labs/lib/stages";
import type {
  LabPredictionResponse,
  LabRunError,
  LabRunResult,
  SourceResolveResult,
} from "@/lib/ml-labs/types";

const DEFAULT_INTENT =
  "Build a trustworthy machine-learning model, show the evidence, and package the outcome for reuse.";
const DEMO_DATASET_PATH = "/data/demo-churn.csv";
const DEMO_TARGET = "churn";

export function MlLabsWorkbench() {
  const [sourceMode, setSourceMode] = useState<SourceMode>("kaggle");
  const [phase, setPhase] = useState<ShellPhase>("idle");
  const [intentPrompt, setIntentPrompt] = useState(DEFAULT_INTENT);
  const [kaggleInput, setKaggleInput] = useState(
    'path = kagglehub.dataset_download("waddahali/kaggle-competition-graph-dataset")',
  );
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedCandidatePath, setSelectedCandidatePath] = useState("");
  const [sourceResolution, setSourceResolution] = useState<SourceResolveResult | null>(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [messages, setMessages] = useState<ShellMessage[]>([]);
  const [result, setResult] = useState<LabRunResult | null>(null);
  const [selectedStageId, setSelectedStageId] = useState(STAGES[0].id);
  const [activeArtifactId, setActiveArtifactId] = useState<string | null>(null);
  const [predictionValues, setPredictionValues] = useState<Record<string, string>>({});
  const [predictionResult, setPredictionResult] = useState<LabPredictionResponse | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isResolving, setIsResolving] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const scheduledTimers = useRef<number[]>([]);
  const resolveTimer = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      clearTimers(scheduledTimers.current);
      if (resolveTimer.current) {
        window.clearTimeout(resolveTimer.current);
      }
    };
  }, []);

  useEffect(() => {
    if (sourceMode !== "kaggle" || !kaggleInput.trim()) {
      return;
    }

    if (resolveTimer.current) {
      window.clearTimeout(resolveTimer.current);
    }

    resolveTimer.current = window.setTimeout(() => {
      void resolveKaggleSource();
    }, 650);

    return () => {
      if (resolveTimer.current) {
        window.clearTimeout(resolveTimer.current);
      }
    };
  }, [kaggleInput, selectedCandidatePath, sourceMode]);

  const stageStatusMap = useMemo(() => getStageStatusMap(messages, phase), [messages, phase]);
  const summaryCards = useMemo(() => (result ? buildSummaryCards(result) : []), [result]);
  const artifacts = useMemo(() => downloadableArtifacts(result), [result]);
  const stageIntel = useMemo(
    () => buildStageIntel(selectedStageId, result, sourceResolution, messages),
    [messages, result, selectedStageId, sourceResolution],
  );
  const unlockedVisualizations = useMemo(
    () => visibleVisualizations(result, messages, selectedStageId),
    [messages, result, selectedStageId],
  );

  async function handleUploadSelected(file: File | null) {
    setSourceMode("upload");
    setSelectedFile(file);
    setSelectedCandidatePath("");
    setSourceResolution(null);
    setResult(null);
    setPredictionResult(null);
    setActiveArtifactId(null);
    if (!file) {
      setTargetColumn("");
      setMessages([]);
      setPhase("idle");
      return;
    }

    await resolveUploadSource(file);
  }

  async function handleLoadDemoDataset() {
    setSourceMode("demo");
    setErrorMessage(null);
    const response = await fetch(DEMO_DATASET_PATH);
    const blob = await response.blob();
    const file = new File([blob], "demo-churn.csv", { type: "text/csv" });
    setSelectedFile(file);
    setSelectedCandidatePath("");
    await resolveUploadSource(file, DEMO_TARGET);
  }

  async function resolveUploadSource(file: File, preferredTarget?: string) {
    clearTimers(scheduledTimers.current);
    setPhase("resolving");
    setErrorMessage(null);
    setResult(null);
    setPredictionResult(null);
    setMessages([
      {
        id: "local-intake",
        agent: "Source Intake Agent",
        stageId: "source-intake",
        status: "running",
        message: `Reading ${file.name} and preparing a structured preview.`,
      },
    ]);

    try {
      const formData = new FormData();
      formData.set("file", file, file.name);
      const resolveResult = await postResolveRequest(formData);
      applyResolvedSource(resolveResult, preferredTarget);
    } catch (error) {
      handleShellError(error);
    }
  }

  async function resolveKaggleSource() {
    if (sourceMode !== "kaggle" || !kaggleInput.trim()) {
      return;
    }

    clearTimers(scheduledTimers.current);
    setPhase("resolving");
    setIsResolving(true);
    setErrorMessage(null);
    setMessages([
      {
        id: "kaggle-source-intake",
        agent: "Connector Agent",
        stageId: "source-intake",
        status: "running",
        message: "Parsing Kaggle input and requesting dataset resolution from the backend connector.",
      },
    ]);

    try {
      const formData = new FormData();
      formData.set("kaggleInput", kaggleInput.trim());
      if (selectedCandidatePath.trim()) {
        formData.set("selectedFilePath", selectedCandidatePath.trim());
      }

      const resolveResult = await postResolveRequest(formData);
      applyResolvedSource(resolveResult);
    } catch (error) {
      handleShellError(error);
    } finally {
      setIsResolving(false);
    }
  }

  function applyResolvedSource(resolveResult: SourceResolveResult, preferredTarget?: string) {
    clearTimers(scheduledTimers.current);
    setSourceResolution(resolveResult);
    setResult(null);
    setPredictionResult(null);
    setActiveArtifactId(null);
    setMessages(normalizeResolveMessages(resolveResult));
    setSelectedStageId(resolveResult.selectedFilePath ? "schema-profiling" : "source-resolution");
    setPhase(resolveResult.selectedFilePath ? "resolved" : "error");
    setTargetColumn(preferredTarget ?? pickSuggestedTarget(resolveResult));
    setSelectedCandidatePath(resolveResult.selectedFilePath ?? "");

    if (!resolveResult.selectedFilePath) {
      setErrorMessage("Choose the CSV table you want ML-Labs to analyze.");
      return;
    }

    setErrorMessage(null);
  }

  async function handleRunLab() {
    if (!sourceResolution?.sourceToken || !targetColumn.trim()) {
      return;
    }

    clearTimers(scheduledTimers.current);
    setPhase("running");
    setErrorMessage(null);
    setResult(null);
    setPredictionResult(null);
    setActiveArtifactId(null);
    setSelectedStageId("preprocessing");
    setMessages((current) => [
      ...current,
      {
        id: `runtime-start-${current.length + 1}`,
        agent: "Run Controller",
        stageId: "preprocessing",
        status: "running",
        message: `Launching the lab on target "${targetColumn}" with the resolved source token.`,
      },
    ]);

    try {
      const formData = new FormData();
      formData.set("sourceToken", sourceResolution.sourceToken);
      formData.set("targetColumn", targetColumn.trim());
      formData.set("intentPrompt", intentPrompt.trim());

      const response = await fetch("/api/lab/run", {
        method: "POST",
        body: formData,
      });
      const payload = (await response.json()) as LabRunResult | LabRunError;
      if (!response.ok) {
        throw new Error(
          (payload as LabRunError).details ?? (payload as LabRunError).error ?? "Run failed.",
        );
      }

      const runResult = payload as LabRunResult;
      setResult(runResult);
      setPredictionValues(buildPredictionSeed(runResult.predictionInputSchema?.fields ?? []));
      setActiveArtifactId(runResult.artifacts[0]?.filename ?? null);
      replayRunTrace(runResult);
    } catch (error) {
      handleShellError(error);
    }
  }

  function replayRunTrace(runResult: LabRunResult) {
    const runMessages = normalizeRunMessages(runResult);
    const baseMessages = sourceResolution ? normalizeResolveMessages(sourceResolution) : [];
    setMessages(baseMessages);

    runMessages.forEach((message, index) => {
      const delay = 220 * (index + 1);
      schedule(() => {
        setMessages((current) => [...current, message]);
        setSelectedStageId(message.stageId);
      }, delay, scheduledTimers.current);
    });

    schedule(() => {
      setPhase("complete");
      setSelectedStageId("export");
    }, runMessages.length * 220 + 280, scheduledTimers.current);
  }

  async function handlePredict() {
    if (!result?.predictionInputSchema) {
      return;
    }

    setIsPredicting(true);
    setErrorMessage(null);
    try {
      const response = await fetch("/api/lab/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          runId: result.runId,
          input: coercePredictionPayload(result.predictionInputSchema.fields, predictionValues),
        }),
      });
      const payload = (await response.json()) as LabPredictionResponse | LabRunError;
      if (!response.ok) {
        throw new Error(
          (payload as LabRunError).details ?? (payload as LabRunError).error ?? "Prediction failed.",
        );
      }

      setPredictionResult(payload as LabPredictionResponse);
    } catch (error) {
      handleShellError(error);
    } finally {
      setIsPredicting(false);
    }
  }

  async function handleDownloadBundle() {
    if (!result) {
      return;
    }

    const JSZip = (await import("jszip")).default;
    const zip = new JSZip();
    artifacts.forEach((artifact) => {
      zip.file(artifact.filename, artifact.content);
    });
    zip.file("plain-english-summary.txt", summaryToText(result.plainEnglishSummary));
    zip.file("run.json", JSON.stringify(result, null, 2));
    const blob = await zip.generateAsync({ type: "blob" });
    downloadBlob(artifactBundleName(result), blob);
  }

  function handleDownloadArtifact(artifact: { filename: string; content: string }) {
    downloadBlob(artifact.filename, new Blob([artifact.content], { type: "text/plain;charset=utf-8" }));
  }

  function handleShellError(error: unknown) {
    clearTimers(scheduledTimers.current);
    const details = error instanceof Error ? error.message : "The lab hit an unexpected error.";
    setPhase("error");
    setErrorMessage(details);
    setMessages((current) => [
      ...current,
      {
        id: `shell-error-${current.length + 1}`,
        agent: "Shell",
        stageId: selectedStageId,
        status: "failed",
        message: details,
      },
    ]);
  }

  const candidateOptions = sourceResolution ? buildCandidateOptions(sourceResolution) : [];
  const canRun = phase !== "running" && Boolean(sourceResolution?.sourceToken && targetColumn.trim());

  return (
    <main className="ml-command-shell">
      <header className="shell-header">
        <div className="brand-lockup">
          <span className="shell-kicker">ML-Labs</span>
          <h1>Autonomous Machine Learning Command Shell</h1>
          <p>
            Paste a Kaggle snippet, URL, or slug. Resolve the dataset, choose what to predict, and
            let the lab surface the models, evidence, critique, and reusable code.
          </p>
        </div>

        <div className="header-metrics">
          <div className="header-chip">
            <span>Status</span>
            <strong>{resolvePhaseLabel(phase, sourceMode)}</strong>
          </div>
          <div className="header-chip">
            <span>Source</span>
            <strong>{sourceResolution?.sourceLabel ?? "waiting"}</strong>
          </div>
          <div className="header-chip">
            <span>Solver</span>
            <strong>
              {result
                ? `${humanizeTaskSubtype(result.problemFraming.taskSubtype)} · ${result.bestModel.modelName}`
                : "pending"}
            </strong>
          </div>
        </div>
      </header>

      <section className="command-grid-top">
        <aside className="command-card source-card">
          <div className="card-header">
            <span className="shell-kicker">Source Console</span>
            <h2>Resolve the dataset first</h2>
          </div>

          <div className="mode-toggle">
            {(["kaggle", "upload", "demo"] as SourceMode[]).map((mode) => (
              <button
                key={mode}
                type="button"
                className={sourceMode === mode ? "shell-button active" : "shell-button"}
                onClick={() => {
                  setSourceMode(mode);
                  setErrorMessage(null);
                  setResult(null);
                  setSourceResolution(null);
                  setMessages([]);
                  setPhase("idle");
                  setTargetColumn(mode === "demo" ? DEMO_TARGET : "");
                }}
              >
                {mode}
              </button>
            ))}
          </div>

          <label className="shell-field">
            <span>Research intent</span>
            <textarea
              rows={4}
              value={intentPrompt}
              onChange={(event) => setIntentPrompt(event.target.value)}
              placeholder="Describe the outcome you want from the lab run."
            />
          </label>

          {sourceMode === "kaggle" ? (
            <label className="shell-field">
              <span>Kaggle link, slug, or code snippet</span>
              <textarea
                rows={6}
                value={kaggleInput}
                onChange={(event) => {
                  setKaggleInput(event.target.value);
                  setSelectedCandidatePath("");
                  setSourceResolution(null);
                  setResult(null);
                  setPredictionResult(null);
                  setPhase("idle");
                }}
                placeholder='Paste something like path = kagglehub.dataset_download("owner/dataset")'
              />
            </label>
          ) : null}

          {sourceMode === "upload" ? (
            <label className="shell-field">
              <span>Upload CSV</span>
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={(event) => void handleUploadSelected(event.target.files?.[0] ?? null)}
              />
            </label>
          ) : null}

          {sourceMode === "demo" ? (
            <button type="button" className="shell-button primary" onClick={() => void handleLoadDemoDataset()}>
              Load bundled demo dataset
            </button>
          ) : null}

          {candidateOptions.length > 1 ? (
            <label className="shell-field">
              <span>Dataset table</span>
              <select
                value={selectedCandidatePath}
                onChange={(event) => setSelectedCandidatePath(event.target.value)}
              >
                <option value="">Choose the CSV table to analyze</option>
                {candidateOptions.map((option) => (
                  <option key={option.path} value={option.path}>
                    {option.label}{option.meta ? ` · ${option.meta}` : ""}
                  </option>
                ))}
              </select>
            </label>
          ) : null}

          <label className="shell-field">
            <span>What should we predict?</span>
            <select
              value={targetColumn}
              onChange={(event) => setTargetColumn(event.target.value)}
              disabled={!sourceResolution?.headers.length}
            >
              <option value="">Choose a prediction target</option>
              {sourceResolution?.headers.map((header) => (
                <option key={header} value={header}>
                  {header}
                </option>
              ))}
            </select>
          </label>

          {sourceResolution?.targetSuggestions.length ? (
            <div className="suggestion-stack">
              {sourceResolution.targetSuggestions.map((suggestion) => (
                <button
                  key={suggestion.column}
                  type="button"
                  className={
                    suggestion.column === targetColumn ? "suggestion-pill active" : "suggestion-pill"
                  }
                  onClick={() => setTargetColumn(suggestion.column)}
                >
                  {suggestion.column} · {Math.round(suggestion.confidence * 100)}%
                </button>
              ))}
            </div>
          ) : null}

          <div className="source-footer">
            <div className="source-detail">
              <span>Resolved table</span>
              <strong>{sourceResolution?.selectedFilePath ?? "pending"}</strong>
            </div>
            <div className="source-detail">
              <span>Headers</span>
              <strong>{sourceResolution?.headers.length ?? 0}</strong>
            </div>
          </div>

          <button
            type="button"
            className="shell-button primary"
            disabled={!canRun || isResolving}
            onClick={() => void handleRunLab()}
          >
            {phase === "running" ? "Running ..." : "Run ML-Labs"}
          </button>
        </aside>

        <section className="command-card topology-card">
          <div className="card-header">
            <span className="shell-kicker">Topology</span>
            <h2>How the ML is happening</h2>
          </div>
          <TopologyMatrix
            selectedStageId={selectedStageId}
            stageStatusMap={stageStatusMap}
            onSelectStage={setSelectedStageId}
          />
        </section>

        <aside className="command-card intel-card">
          <div className="card-header">
            <span className="shell-kicker">Stage Intel</span>
            <h2>{stageIntel.title}</h2>
          </div>
          <p className="intel-copy">{stageIntel.description}</p>
          <div className="intel-highlight">
            <strong>{stageIntel.callout}</strong>
          </div>
          <div className="intel-stack">
            {stageIntel.details.map((detail) => (
              <p key={detail}>{detail}</p>
            ))}
          </div>
          {result ? (
            <div className="framing-block">
              <span className="shell-kicker">Problem framing</span>
              <h3>{humanizeTaskSubtype(result.problemFraming.taskSubtype)}</h3>
              <p>{result.problemFraming.rationale}</p>
              <div className="framing-metrics">
                <span>{result.problemFraming.primaryMetric}</span>
                <span>{result.bestModel.modelName}</span>
              </div>
            </div>
          ) : null}
        </aside>
      </section>

      <section className="command-grid-middle">
        <article className="command-card trace-card">
          <div className="card-header">
            <span className="shell-kicker">Assistant Rail</span>
            <h2>Resolution and run trace</h2>
          </div>
          <div className="trace-list" aria-live="polite">
            {messages.length ? (
              messages.map((message) => (
                <article key={message.id} className={`trace-item tone-${message.status}`}>
                  <span>{message.agent}</span>
                  <strong>{message.message}</strong>
                </article>
              ))
            ) : (
              <article className="trace-item tone-queued">
                <span>Shell</span>
                <strong>Paste a Kaggle reference or load a CSV to begin.</strong>
              </article>
            )}
          </div>
        </article>

        <article className="command-card preview-card">
          <div className="card-header">
            <span className="shell-kicker">Resolved Dataset</span>
            <h2>Preview and target discovery</h2>
          </div>
          {sourceResolution?.previewRows.length ? (
            <div className="preview-table">
              <div
                className="preview-head"
                style={{ gridTemplateColumns: `repeat(${sourceResolution.headers.length}, minmax(0, 1fr))` }}
              >
                {sourceResolution.headers.map((header) => (
                  <span key={header}>{header}</span>
                ))}
              </div>
              {sourceResolution.previewRows.slice(0, 10).map((row, rowIndex) => (
                <div
                  key={`row-${rowIndex}`}
                  className="preview-row"
                  style={{ gridTemplateColumns: `repeat(${sourceResolution.headers.length}, minmax(0, 1fr))` }}
                >
                  {row.map((cell, cellIndex) => (
                    <span key={`${rowIndex}-${cellIndex}`}>{cell}</span>
                  ))}
                </div>
              ))}
            </div>
          ) : (
            <p className="empty-message">
              Once the source is resolved, a real preview of the chosen CSV table appears here.
            </p>
          )}
        </article>
      </section>

      <EvidenceWorkspace
        runResult={result}
        summaryCards={summaryCards}
        visualizations={unlockedVisualizations}
        artifacts={artifacts}
        activeArtifactId={activeArtifactId}
        onSelectArtifact={setActiveArtifactId}
        onDownloadArtifact={handleDownloadArtifact}
        onDownloadBundle={() => void handleDownloadBundle()}
        predictionValues={predictionValues}
        predictionResult={predictionResult}
        isPredicting={isPredicting}
        onPredictionChange={(fieldName, value) =>
          setPredictionValues((current) => ({ ...current, [fieldName]: value }))
        }
        onPredict={() => void handlePredict()}
      />

      {errorMessage ? (
        <section className="shell-error-banner">
          <span className="shell-kicker">Shell error</span>
          <p>{errorMessage}</p>
        </section>
      ) : null}
    </main>
  );
}

async function postResolveRequest(formData: FormData): Promise<SourceResolveResult> {
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

function resolvePhaseLabel(phase: ShellPhase, sourceMode: SourceMode): string {
  if (phase === "resolving") {
    return sourceMode === "kaggle" ? "resolving Kaggle ..." : "resolving dataset ...";
  }
  if (phase === "resolved") {
    return "ready to run";
  }
  if (phase === "running") {
    return "training models ...";
  }
  if (phase === "complete") {
    return "run complete";
  }
  if (phase === "error") {
    return "attention needed";
  }
  return "waiting for source";
}

function schedule(callback: () => void, delayMs: number, bucket: number[]) {
  const handle = window.setTimeout(callback, delayMs);
  bucket.push(handle);
}

function clearTimers(handles: number[]) {
  handles.forEach((handle) => window.clearTimeout(handle));
  handles.length = 0;
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
