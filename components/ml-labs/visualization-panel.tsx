import type { VisualizationViewModel } from "@/lib/ml-labs/workbench-normalizer";

type VisualizationDeckProps = {
  visualizations: VisualizationViewModel[];
};

type PlotPoint = {
  x: number;
  y: number;
};

export function VisualizationDeck({ visualizations }: VisualizationDeckProps) {
  return (
    <section className="result-section">
      <div className="section-heading">
        <span className="section-kicker">Visual Layer</span>
        <h3>Evaluation evidence</h3>
      </div>

      <div className="visualization-grid">
        {visualizations.map((visualization) => (
          <article key={visualization.id} className="visualization-card">
            <header className="visualization-header">
              <span className="visualization-chip">{visualization.type.replaceAll("_", " ")}</span>
              <h4>{visualization.title}</h4>
            </header>
            <VisualizationSurface visualization={visualization} />
          </article>
        ))}
      </div>
    </section>
  );
}

function VisualizationSurface({ visualization }: { visualization: VisualizationViewModel }) {
  switch (visualization.variant) {
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
        <div className="visualization-fallback">
          <p>{JSON.stringify(visualization.data, null, 2)}</p>
        </div>
      );
  }
}

function BarVisualization({ visualization }: { visualization: VisualizationViewModel }) {
  const rows = Array.isArray(visualization.data) ? visualization.data.slice(0, 8) : [];

  return (
    <div className="bar-stack">
      {rows.map((row, index) => {
        const label = resolveBarLabel(row);
        const numericValue = resolveBarValue(row);
        const width = Math.max(8, Math.min(100, numericValue * 100));

        return (
          <div key={`${label}-${index}`} className="bar-row">
            <div className="bar-row-copy">
              <span>{label}</span>
              <strong>{formatNumber(numericValue)}</strong>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${width}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function GridVisualization({ visualization }: { visualization: VisualizationViewModel }) {
  const data = visualization.data as
    | { columns?: string[]; matrix?: number[][]; labels?: string[] }
    | undefined;

  const labels = data?.columns ?? data?.labels ?? [];
  const matrix = data?.matrix ?? [];

  if (!labels.length || !matrix.length) {
    return <p className="visualization-empty">No grid data available for this run.</p>;
  }

  return (
    <div className="grid-heatmap">
      <div
        className="grid-heatmap-matrix"
        style={{ gridTemplateColumns: `repeat(${labels.length}, minmax(0, 1fr))` }}
      >
        {matrix.flatMap((row, rowIndex) =>
          row.map((value, columnIndex) => (
            <div
              key={`${rowIndex}-${columnIndex}`}
              className="grid-cell"
              style={{ opacity: 0.2 + Math.min(Math.abs(value), 1) * 0.8 }}
              title={`${labels[rowIndex] ?? rowIndex} · ${labels[columnIndex] ?? columnIndex}: ${value}`}
            >
              <span>{formatNumber(value)}</span>
            </div>
          )),
        )}
      </div>
      <div className="grid-labels">
        {labels.slice(0, 8).map((label) => (
          <span key={label}>{label}</span>
        ))}
      </div>
    </div>
  );
}

function LineVisualization({ visualization }: { visualization: VisualizationViewModel }) {
  const points = resolveLinePoints(visualization);
  if (!points.length) {
    return <p className="visualization-empty">No line data available for this run.</p>;
  }

  const path = points
    .map((point, index) => {
      const x = 10 + point.x * 240;
      const y = 150 - point.y * 120;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

  return (
    <div className="line-plot">
      <svg viewBox="0 0 260 160" className="line-plot-svg" role="img" aria-label={visualization.title}>
        <path d="M10 150 H250" className="line-axis" />
        <path d="M10 150 V20" className="line-axis" />
        <path d={path} className="line-path" />
      </svg>
      <div className="line-caption">
        <span>{formatNumber(points[0]?.y ?? 0)}</span>
        <span>{formatNumber(points[points.length - 1]?.y ?? 0)}</span>
      </div>
    </div>
  );
}

function GraphVisualization({ visualization }: { visualization: VisualizationViewModel }) {
  const data = visualization.data as { nodes?: string[]; edges?: string[][] } | undefined;
  const nodes = data?.nodes ?? [];
  const edges = data?.edges ?? [];

  return (
    <div className="graph-rail">
      <div className="graph-nodes">
        {nodes.map((node) => (
          <div key={node} className="graph-node">
            <span>{node}</span>
          </div>
        ))}
      </div>
      <div className="graph-edges">
        {edges.slice(0, 8).map(([from, to], index) => (
          <div key={`${from}-${to}-${index}`} className="graph-edge">
            <span>{from}</span>
            <i />
            <span>{to}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function resolveBarLabel(row: unknown): string {
  if (!row || typeof row !== "object") {
    return "value";
  }

  const record = row as Record<string, unknown>;
  return String(
    record.label ??
      record.modelName ??
      record.feature ??
      record.column ??
      record.sourceColumn ??
      "value",
  );
}

function resolveBarValue(row: unknown): number {
  if (!row || typeof row !== "object") {
    return 0;
  }

  const record = row as Record<string, unknown>;
  const candidate =
    record.ratio ??
    record.score ??
    record.importance ??
    record.missingRatio ??
    record.count ??
    0;

  const numeric = Number(candidate);
  if (Number.isNaN(numeric)) {
    return 0;
  }

  return numeric > 1 && numeric <= 100 ? numeric / 100 : numeric;
}

function resolveLinePoints(visualization: VisualizationViewModel): PlotPoint[] {
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
    return rows.slice(0, 32).map((row, index) => {
      const record = row as Record<string, unknown>;
      return {
        x: normalize(index, 0, Math.max(rows.length - 1, 1)),
        y: normalize(Number(record.residual ?? 0), -5000, 5000),
      };
    });
  }

  const rows = Array.isArray(visualization.data) ? visualization.data : [];
  return rows.slice(0, 32).map((row) => {
    const record = row as Record<string, unknown>;
    return {
      x: normalize(Number(record.actual ?? 0), 0, 100000),
      y: normalize(Number(record.predicted ?? 0), 0, 100000),
    };
  });
}

function pairArrays(left: number[], right: number[]): PlotPoint[] {
  return left.slice(0, Math.min(left.length, right.length)).map((value, index) => ({
    x: clamp01(value),
    y: clamp01(right[index] ?? 0),
  }));
}

function normalize(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }

  const span = max - min;
  if (span <= 0) {
    return 0;
  }

  return clamp01((value - min) / span);
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return "0";
  }

  return value.toFixed(value >= 10 ? 1 : 3);
}
