import type { ChangeEvent } from "react";
import type { DatasetPreview, SourceMode } from "@/lib/ml-labs/workbench-normalizer";

type SourcePanelProps = {
  sourceMode: SourceMode;
  preview: DatasetPreview | null;
  intentPrompt: string;
  kaggleReference: string;
  kaggleFilePath: string;
  targetColumn: string;
  statusLabel: string;
  isBusy: boolean;
  canRun: boolean;
  onSourceModeChange: (mode: SourceMode) => void;
  onIntentPromptChange: (value: string) => void;
  onKaggleReferenceChange: (value: string) => void;
  onKaggleFilePathChange: (value: string) => void;
  onTargetColumnChange: (value: string) => void;
  onFileSelected: (file: File | null) => void;
  onLoadDemoCsv: () => void;
  onRun: () => void;
};

const SOURCE_LABELS: Record<SourceMode, string> = {
  upload: "Upload CSV",
  demo: "Demo CSV",
  kaggle: "Kaggle",
};

export function SourcePanel({
  sourceMode,
  preview,
  intentPrompt,
  kaggleReference,
  kaggleFilePath,
  targetColumn,
  statusLabel,
  isBusy,
  canRun,
  onSourceModeChange,
  onIntentPromptChange,
  onKaggleReferenceChange,
  onKaggleFilePathChange,
  onTargetColumnChange,
  onFileSelected,
  onLoadDemoCsv,
  onRun,
}: SourcePanelProps) {
  return (
    <section className="workbench-panel source-panel">
      <div className="panel-header">
        <div>
          <span className="panel-kicker">Panel 01</span>
          <h2>Dataset source</h2>
        </div>
        <span className={`status-pill status-${statusLabel.replace(/\s+/g, "-")}`}>{statusLabel}</span>
      </div>

      <div className="mode-switch">
        {(Object.keys(SOURCE_LABELS) as SourceMode[]).map((mode) => (
          <button
            key={mode}
            type="button"
            className={mode === sourceMode ? "mode-option active" : "mode-option"}
            onClick={() => onSourceModeChange(mode)}
            disabled={isBusy}
          >
            {SOURCE_LABELS[mode]}
          </button>
        ))}
      </div>

      <label className="field">
        <span>Research intent</span>
        <textarea
          rows={3}
          value={intentPrompt}
          onChange={(event) => onIntentPromptChange(event.target.value)}
          placeholder="Describe the outcome you want from the lab run."
        />
      </label>

      {sourceMode === "upload" ? (
        <div className="field-group">
          <label className="field">
            <span>CSV file</span>
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={(event) => handleFileChange(event, onFileSelected)}
              disabled={isBusy}
            />
          </label>
        </div>
      ) : null}

      {sourceMode === "demo" ? (
        <div className="field-group">
          <button type="button" className="primary-action ghost" onClick={onLoadDemoCsv} disabled={isBusy}>
            Load bundled demo CSV
          </button>
          <p className="field-note">
            Uses <span>public/data/demo-churn.csv</span> and auto-fills the churn target.
          </p>
        </div>
      ) : null}

      {sourceMode === "kaggle" ? (
        <div className="field-group">
          <label className="field">
            <span>Kaggle slug or URL</span>
            <input
              type="text"
              value={kaggleReference}
              onChange={(event) => onKaggleReferenceChange(event.target.value)}
              placeholder="waddahali/kaggle-competition-graph-dataset"
            />
          </label>
          <label className="field">
            <span>Optional Kaggle file path</span>
            <input
              type="text"
              value={kaggleFilePath}
              onChange={(event) => onKaggleFilePathChange(event.target.value)}
              placeholder="nodes.csv"
            />
          </label>
        </div>
      ) : null}

      <div className="field-group dual">
        {sourceMode === "kaggle" ? (
          <label className="field">
            <span>Target column</span>
            <input
              type="text"
              value={targetColumn}
              onChange={(event) => onTargetColumnChange(event.target.value)}
              placeholder="type"
            />
          </label>
        ) : (
          <label className="field">
            <span>Target column</span>
            <select
              value={targetColumn}
              onChange={(event) => onTargetColumnChange(event.target.value)}
              disabled={!preview?.headers.length}
            >
              <option value="">Select a column</option>
              {preview?.headers.map((header) => (
                <option key={header} value={header}>
                  {header}
                </option>
              ))}
            </select>
          </label>
        )}

        <div className="source-metadata">
          <span className="source-metadata-label">Current source</span>
          <strong>{preview?.filename ?? (kaggleReference || "waiting for input")}</strong>
          <p>
            {preview
              ? `${preview.headers.length} headers detected · ${preview.rows.length} preview rows loaded`
              : sourceMode === "kaggle"
                ? "Target will be resolved against the backend dataset source."
                : "Load a file to unlock structured profiling."}
          </p>
        </div>
      </div>

      <div className="panel-actions">
        <button type="button" className="primary-action" onClick={onRun} disabled={!canRun || isBusy}>
          Run dataset
        </button>
      </div>
    </section>
  );
}

function handleFileChange(
  event: ChangeEvent<HTMLInputElement>,
  onFileSelected: (file: File | null) => void,
) {
  const file = event.target.files?.[0] ?? null;
  onFileSelected(file);
}
