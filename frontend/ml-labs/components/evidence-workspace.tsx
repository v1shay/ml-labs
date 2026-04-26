"use client";

import type {
  DownloadableArtifact,
  SummaryCard,
} from "@/frontend/ml-labs/lib/stages";
import {
  predictionSummary,
  summaryToText,
  visualizationVariant,
} from "@/frontend/ml-labs/lib/stages";
import type {
  LabPredictionResponse,
  LabRunResult,
  PredictionInputField,
  Visualization,
} from "@/lib/ml-labs/types";

type EvidenceWorkspaceProps = {
  runResult: LabRunResult | null;
  summaryCards: SummaryCard[];
  visualizations: Visualization[];
  artifacts: DownloadableArtifact[];
  activeArtifactId: string | null;
  onSelectArtifact: (artifactId: string) => void;
  onDownloadArtifact: (artifact: DownloadableArtifact) => void;
  onDownloadBundle: () => void;
  predictionValues: Record<string, string>;
  predictionResult: LabPredictionResponse | null;
  isPredicting: boolean;
  onPredictionChange: (fieldName: string, value: string) => void;
  onPredict: () => void;
};

export function EvidenceWorkspace({
  runResult,
  summaryCards,
  visualizations,
  artifacts,
  activeArtifactId,
  onSelectArtifact,
  onDownloadArtifact,
  onDownloadBundle,
  predictionValues,
  predictionResult,
  isPredicting,
  onPredictionChange,
  onPredict,
}: EvidenceWorkspaceProps) {
  const activeArtifact =
    artifacts.find((artifact) => artifact.id === activeArtifactId) ?? artifacts[0] ?? null;

  return (
    <section className="command-lower">
      <article className="command-card summary-card">
        <div className="card-header">
          <span className="shell-kicker">Summary</span>
          <h3>Plain-English outcome</h3>
        </div>

        {runResult ? (
          <div className="summary-layout">
            <div className="summary-grid">
              {summaryCards.map((card) => (
                <div key={card.label} className="metric-tile">
                  <span>{card.label}</span>
                  <strong>{card.value}</strong>
                  <p>{card.hint}</p>
                </div>
              ))}
            </div>

            <div className="narrative-card">
              <h4>{runResult.plainEnglishSummary.headline}</h4>
              <p>{runResult.plainEnglishSummary.shortExplanation}</p>
              <div className="bullet-stack">
                {runResult.plainEnglishSummary.takeaways.map((takeaway) => (
                  <p key={takeaway}>{takeaway}</p>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <EmptyMessage text="Resolve and run a dataset to unlock the summary layer." />
        )}
      </article>

      <div className="command-grid">
        <article className="command-card evidence-card">
          <div className="card-header">
            <span className="shell-kicker">Evidence</span>
            <h3>Graphs and diagnostics</h3>
          </div>
          {visualizations.length ? (
            <div className="visualization-grid">
              {visualizations.map((visualization) => (
                <VisualizationCard key={visualization.id} visualization={visualization} />
              ))}
            </div>
          ) : (
            <EmptyMessage text="Resolved profiling and model evidence will appear here as stages complete." />
          )}
        </article>

        <article className="command-card leaderboard-card-shell">
          <div className="card-header">
            <span className="shell-kicker">Leaderboard</span>
            <h3>Model comparison</h3>
          </div>
          {runResult ? (
            <div className="leaderboard-shell">
              <div className="leaderboard-head">
                <span>Model</span>
                <span>Family</span>
                <span>Metric</span>
                <span>Score</span>
              </div>
              {runResult.leaderboard.map((entry) => (
                <div key={`${entry.modelName}-${entry.family}`} className="leaderboard-row-shell">
                  <span>{entry.modelName}</span>
                  <span>{entry.family}</span>
                  <span>{entry.metricName}</span>
                  <strong>{entry.score.toFixed(3)}</strong>
                </div>
              ))}
            </div>
          ) : (
            <EmptyMessage text="The leaderboard appears after the experiment sweep completes." />
          )}
        </article>

        <article className="command-card artifact-shell">
          <div className="card-header">
            <span className="shell-kicker">Exports</span>
            <h3>Downloadable artifacts</h3>
          </div>

          {runResult ? (
            <>
              <div className="artifact-actions">
                {artifacts.map((artifact) => (
                  <button
                    key={artifact.id}
                    type="button"
                    className={
                      artifact.id === activeArtifact?.id ? "shell-button active" : "shell-button"
                    }
                    onClick={() => onSelectArtifact(artifact.id)}
                  >
                    {artifact.filename}
                  </button>
                ))}
              </div>

              <div className="artifact-downloads">
                {artifacts.map((artifact) => (
                  <button
                    key={`${artifact.id}-download`}
                    type="button"
                    className="shell-button ghost"
                    onClick={() => onDownloadArtifact(artifact)}
                  >
                    Download {artifact.filename}
                  </button>
                ))}
                <button type="button" className="shell-button primary" onClick={onDownloadBundle}>
                  Download run bundle
                </button>
                <button
                  type="button"
                  className="shell-button ghost"
                  onClick={() => {
                    if (runResult) {
                      onDownloadArtifact({
                        id: "plain-english-summary.txt",
                        filename: "plain-english-summary.txt",
                        type: "report",
                        content: summaryToText(runResult.plainEnglishSummary),
                      });
                    }
                  }}
                >
                  Download summary
                </button>
              </div>

              <div className="artifact-preview">
                <pre>{activeArtifact?.content ?? "Choose an artifact to preview it here."}</pre>
              </div>
            </>
          ) : (
            <EmptyMessage text="Reports, code, and the run bundle appear after a successful experiment." />
          )}
        </article>

        <article className="command-card report-shell">
          <div className="card-header">
            <span className="shell-kicker">Research Layer</span>
            <h3>Critique and report</h3>
          </div>
          {runResult ? (
            <div className="report-stack">
              <section className="critic-grid-shell">
                {[
                  { title: "Warnings", items: runResult.criticReport.warnings },
                  { title: "Failure Modes", items: runResult.criticReport.failureModes },
                  { title: "Next Experiments", items: runResult.criticReport.nextExperiments },
                  { title: "Limitations", items: runResult.criticReport.limitations },
                ].map(({ title, items }) => (
                  <article key={title} className="critic-card-shell">
                    <span className="shell-kicker">{title}</span>
                    {items.length ? (
                      items.map((item) => <p key={item}>{item}</p>)
                    ) : (
                      <p>No major notes in this category.</p>
                    )}
                  </article>
                ))}
              </section>
              <div className="report-preview">
                <pre>{runResult.finalReportMarkdown}</pre>
              </div>
            </div>
          ) : (
            <EmptyMessage text="The report layer activates after the backend packaging stage completes." />
          )}
        </article>

        <article className="command-card prediction-shell">
          <div className="card-header">
            <span className="shell-kicker">Try Here</span>
            <h3>Score one fresh example</h3>
          </div>
          {runResult?.predictionInputSchema ? (
            <div className="prediction-layout">
              <div className="prediction-fields">
                {runResult.predictionInputSchema.fields.map((field) => (
                  <PredictionField
                    key={field.name}
                    field={field}
                    value={predictionValues[field.name] ?? ""}
                    onChange={(value) => onPredictionChange(field.name, value)}
                  />
                ))}
              </div>
              <div className="prediction-output">
                <p>{predictionSummary(predictionResult)}</p>
                {predictionResult ? (
                  <>
                    <strong>{predictionResult.explanation}</strong>
                    <div className="bullet-stack">
                      {predictionResult.topFactors.map((factor) => (
                        <p key={factor}>{factor}</p>
                      ))}
                    </div>
                  </>
                ) : null}
                <button
                  type="button"
                  className="shell-button primary"
                  onClick={onPredict}
                  disabled={isPredicting}
                >
                  {isPredicting ? "Scoring ..." : "Predict with saved model"}
                </button>
              </div>
            </div>
          ) : (
            <EmptyMessage text="The prediction form appears after the run returns a scoring schema." />
          )}
        </article>
      </div>
    </section>
  );
}

function PredictionField({
  field,
  value,
  onChange,
}: {
  field: PredictionInputField;
  value: string;
  onChange: (value: string) => void;
}) {
  if (field.options?.length) {
    return (
      <label className="shell-field">
        <span>{field.label}</span>
        <select value={value} onChange={(event) => onChange(event.target.value)}>
          {field.options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>
    );
  }

  return (
    <label className="shell-field">
      <span>{field.label}</span>
      <input
        type={field.kind === "number" ? "number" : "text"}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={field.example !== undefined ? String(field.example) : ""}
      />
    </label>
  );
}

function VisualizationCard({ visualization }: { visualization: Visualization }) {
  return (
    <article className="viz-card">
      <div className="viz-header">
        <span className="shell-kicker">{visualization.type.replaceAll("_", " ")}</span>
        <h4>{visualization.title}</h4>
      </div>
      <VisualizationSurface visualization={visualization} />
    </article>
  );
}

function VisualizationSurface({ visualization }: { visualization: Visualization }) {
  const variant = visualizationVariant(visualization.type);
  switch (variant) {
    case "bars":
      return <BarVisualization visualization={visualization} />;
    case "grid":
      return <GridVisualization visualization={visualization} />;
    case "line":
      return <LineVisualization visualization={visualization} />;
    case "graph":
      return <GraphVisualization visualization={visualization} />;
    default:
      return (
        <div className="viz-fallback">
          <pre>{JSON.stringify(visualization.data, null, 2)}</pre>
        </div>
      );
  }
}

function BarVisualization({ visualization }: { visualization: Visualization }) {
  const rows = Array.isArray(visualization.data) ? visualization.data.slice(0, 10) : [];
  return (
    <div className="bar-stack-shell">
      {rows.map((row, index) => {
        const record = row as Record<string, unknown>;
        const label = String(
          record.label ??
            record.modelName ??
            record.feature ??
            record.column ??
            record.sourceColumn ??
            `value-${index + 1}`,
        );
        const numericValue = Number(
          record.ratio ??
            record.score ??
            record.importance ??
            record.missingRatio ??
            record.count ??
            0,
        );
        const normalized = Number.isFinite(numericValue)
          ? numericValue > 1 && numericValue <= 100
            ? numericValue / 100
            : numericValue
          : 0;

        return (
          <div key={`${label}-${index}`} className="bar-row-shell">
            <div className="bar-label">
              <span>{label}</span>
              <strong>{Number.isFinite(numericValue) ? numericValue.toFixed(3) : "0.000"}</strong>
            </div>
            <div className="bar-track-shell">
              <div
                className="bar-fill-shell"
                style={{ width: `${Math.max(8, Math.min(100, normalized * 100))}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function GridVisualization({ visualization }: { visualization: Visualization }) {
  const data = visualization.data as
    | { columns?: string[]; matrix?: number[][]; labels?: string[] }
    | undefined;
  const labels = data?.columns ?? data?.labels ?? [];
  const matrix = data?.matrix ?? [];

  if (!labels.length || !matrix.length) {
    return <EmptyMessage text="No grid data is available for this stage." compact />;
  }

  return (
    <div className="grid-shell">
      <div
        className="grid-shell-matrix"
        style={{ gridTemplateColumns: `repeat(${labels.length}, minmax(0, 1fr))` }}
      >
        {matrix.flatMap((row, rowIndex) =>
          row.map((value, columnIndex) => (
            <div
              key={`${rowIndex}-${columnIndex}`}
              className="grid-cell-shell"
              style={{ opacity: 0.22 + Math.min(Math.abs(Number(value)), 1) * 0.78 }}
              title={`${labels[rowIndex] ?? rowIndex} · ${labels[columnIndex] ?? columnIndex}: ${value}`}
            />
          )),
        )}
      </div>
      <div className="grid-label-row">
        {labels.slice(0, 8).map((label) => (
          <span key={label}>{label}</span>
        ))}
      </div>
    </div>
  );
}

function LineVisualization({ visualization }: { visualization: Visualization }) {
  const points = resolveLinePoints(visualization);
  if (!points.length) {
    return <EmptyMessage text="No plotted points are available for this stage." compact />;
  }

  const path = points
    .map((point, index) => {
      const x = 16 + point.x * 250;
      const y = 156 - point.y * 124;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

  return (
    <div className="line-shell">
      <svg viewBox="0 0 282 170" className="line-shell-svg" role="img" aria-label={visualization.title}>
        <path d="M16 156 H268" className="axis-path" />
        <path d="M16 156 V24" className="axis-path" />
        <path d={path} className="signal-path" />
      </svg>
    </div>
  );
}

function GraphVisualization({ visualization }: { visualization: Visualization }) {
  const data = visualization.data as { nodes?: string[]; edges?: string[][] } | undefined;
  const nodes = data?.nodes ?? [];
  const edges = data?.edges ?? [];
  return (
    <div className="mini-graph-shell">
      <div className="mini-node-row">
        {nodes.map((node) => (
          <span key={node} className="mini-node-pill">
            {node}
          </span>
        ))}
      </div>
      <div className="mini-edge-row">
        {edges.slice(0, 10).map(([from, to], index) => (
          <p key={`${from}-${to}-${index}`}>{from} → {to}</p>
        ))}
      </div>
    </div>
  );
}

function resolveLinePoints(visualization: Visualization): Array<{ x: number; y: number }> {
  if (visualization.type === "roc_curve") {
    const data = visualization.data as { fpr?: number[]; tpr?: number[] } | undefined;
    return pairArrays(data?.fpr ?? [], data?.tpr ?? []);
  }

  if (visualization.type === "pr_curve") {
    const data = visualization.data as { recall?: number[]; precision?: number[] } | undefined;
    return pairArrays(data?.recall ?? [], data?.precision ?? []);
  }

  if (visualization.type === "residual_plot") {
    const rows = Array.isArray(visualization.data) ? visualization.data : [];
    return rows.slice(0, 36).map((row, index) => {
      const record = row as Record<string, unknown>;
      return {
        x: normalize(index, 0, Math.max(rows.length - 1, 1)),
        y: normalize(Number(record.residual ?? 0), -5000, 5000),
      };
    });
  }

  const rows = Array.isArray(visualization.data) ? visualization.data : [];
  return rows.slice(0, 36).map((row) => {
    const record = row as Record<string, unknown>;
    return {
      x: normalize(Number(record.actual ?? 0), 0, 100000),
      y: normalize(Number(record.predicted ?? 0), 0, 100000),
    };
  });
}

function pairArrays(left: number[], right: number[]) {
  return left.slice(0, Math.min(left.length, right.length)).map((value, index) => ({
    x: clamp01(value),
    y: clamp01(right[index] ?? 0),
  }));
}

function normalize(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) {
    return 0;
  }

  const span = max - min;
  if (span <= 0) {
    return 0;
  }

  return clamp01((value - min) / span);
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function EmptyMessage({ text, compact = false }: { text: string; compact?: boolean }) {
  return <p className={compact ? "empty-message compact" : "empty-message"}>{text}</p>;
}
