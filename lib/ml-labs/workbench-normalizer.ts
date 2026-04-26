import type {
  AgentTraceItem,
  ArtifactType,
  LabArtifact,
  LabRunResult,
  LeaderboardEntry,
  Visualization,
  VisualizationType,
} from "@/lib/ml-labs/types";

export type SourceMode = "upload" | "demo" | "kaggle";

export type DatasetPreview = {
  filename: string;
  headers: string[];
  rows: string[][];
};

export type WorkbenchMessage = {
  id: string;
  label: string;
  text: string;
  tone: "neutral" | "accent" | "success" | "warning";
};

export type DatasetMetricCard = {
  label: string;
  value: string;
  hint: string;
};

export type ArtifactTab = {
  id: string;
  label: string;
  type: ArtifactType;
  content: string;
};

export type VisualizationViewModel = {
  id: string;
  title: string;
  type: VisualizationType;
  variant: "bars" | "grid" | "line" | "graph" | "fallback";
  data: unknown;
};

export function buildDatasetMetricCards(result: LabRunResult): DatasetMetricCard[] {
  const { datasetProfile, bestModel, predictionInputSchema, visualizations, artifacts } = result;

  return [
    {
      label: "Rows",
      value: datasetProfile.rows.toLocaleString(),
      hint: "Detected during intake.",
    },
    {
      label: "Columns",
      value: datasetProfile.columns.toString(),
      hint: `${datasetProfile.numericColumns.length} numeric / ${datasetProfile.categoricalColumns.length} categorical`,
    },
    {
      label: "Target",
      value: datasetProfile.targetColumn,
      hint: datasetProfile.problemType,
    },
    {
      label: "Best Model",
      value: bestModel.modelName,
      hint: `${bestModel.metricName} ${bestModel.score.toFixed(3)}`,
    },
    {
      label: "Improvement",
      value: `${bestModel.absoluteImprovement.toFixed(3)}`,
      hint: `${bestModel.relativeImprovement.toFixed(1)}% over baseline`,
    },
    {
      label: "Schema Fields",
      value: `${predictionInputSchema?.fields.length ?? 0}`,
      hint: `${visualizations.length} visualizations / ${artifacts.length} artifacts`,
    },
  ];
}

export function buildSchemaStreamLines(result: LabRunResult): string[] {
  const leaderboardWinner = result.leaderboard[0];

  return [
    "{",
    `  "runId": "${result.runId}",`,
    `  "problemType": "${result.datasetProfile.problemType}",`,
    `  "targetColumn": "${result.datasetProfile.targetColumn}",`,
    `  "rows": ${result.datasetProfile.rows},`,
    `  "columns": ${result.datasetProfile.columns},`,
    `  "numericColumns": ${result.datasetProfile.numericColumns.length},`,
    `  "categoricalColumns": ${result.datasetProfile.categoricalColumns.length},`,
    `  "leaderboardWinner": "${leaderboardWinner?.modelName ?? "n/a"}",`,
    `  "winnerScore": ${leaderboardWinner?.score?.toFixed(3) ?? "0.000"},`,
    `  "visualizationCount": ${result.visualizations.length},`,
    `  "artifactCount": ${result.artifacts.length},`,
    `  "schemaFieldCount": ${result.predictionInputSchema?.fields.length ?? 0}`,
    "}",
  ];
}

export function buildTraceReplay(result: LabRunResult): WorkbenchMessage[] {
  return result.agentTrace.map((item, index) => ({
    id: `${item.stageId}-${index}`,
    label: item.agent,
    text: item.message,
    tone: traceTone(item),
  }));
}

export function buildArtifactTabs(result: LabRunResult): ArtifactTab[] {
  return result.artifacts
    .filter((artifact) => typeof artifact.content === "string" && artifact.content.length > 0)
    .map((artifact) => ({
      id: artifact.filename,
      label: artifact.filename,
      type: artifact.type,
      content: artifact.content ?? "",
    }));
}

export function buildVisualizationViewModels(result: LabRunResult): VisualizationViewModel[] {
  return result.visualizations.map((visualization) => ({
    id: visualization.id,
    title: visualization.title,
    type: visualization.type,
    variant: visualizationVariant(visualization.type),
    data: visualization.data,
  }));
}

export function buildLeaderboardRows(result: LabRunResult): LeaderboardEntry[] {
  return result.leaderboard;
}

export function buildCriticBlocks(result: LabRunResult): Array<{
  title: string;
  items: string[];
}> {
  return [
    { title: "Warnings", items: result.criticReport.warnings },
    { title: "Failure Modes", items: result.criticReport.failureModes },
    { title: "Next Experiments", items: result.criticReport.nextExperiments },
    { title: "Limitations", items: result.criticReport.limitations },
  ];
}

export function buildInitialMessages(sourceMode: SourceMode): WorkbenchMessage[] {
  if (sourceMode === "kaggle") {
    return [
      message("source-search", "Connector", "searching dataset registry ...", "accent"),
      message("source-search-2", "Planner", "mapping Kaggle source into the run contract.", "neutral"),
    ];
  }

  return [
    message("source-ingest", "Reader", "ingesting source file ...", "accent"),
    message("source-ingest-2", "Profiler", "preparing row preview and header scan.", "neutral"),
  ];
}

export function buildAnalyzingMessages(sourceMode: SourceMode): WorkbenchMessage[] {
  const common = [
    message("analyze-1", "Reader", "reading headers", "neutral"),
    message("analyze-2", "Framer", "inferring target behavior", "neutral"),
    message("analyze-3", "Profiler", "profiling schema", "neutral"),
    message("analyze-4", "Runner", "assembling run payload", "neutral"),
    message("analyze-5", "Runner", "waiting for experiment engine", "accent"),
  ];

  if (sourceMode === "kaggle") {
    return [
      message("kaggle-1", "Connector", "searching dataset source ...", "accent"),
      message("kaggle-2", "Connector", "resolving slug and candidate CSV files", "neutral"),
      ...common,
    ];
  }

  return common;
}

export function previewRowsToLines(preview: DatasetPreview): string[] {
  return preview.rows.map((row, index) => {
    const pairs = preview.headers.map((header, cellIndex) => `${header}=${row[cellIndex] ?? ""}`);
    return `${String(index + 1).padStart(2, "0")}  ${pairs.join("  ·  ")}`;
  });
}

export function resolveArtifactLabel(artifact: LabArtifact): string {
  return artifact.filename.replace(/\.[^.]+$/, "");
}

function traceTone(item: AgentTraceItem): WorkbenchMessage["tone"] {
  if (item.status === "failed") {
    return "warning";
  }

  if (item.status === "warning") {
    return "warning";
  }

  if (item.status === "complete") {
    return "success";
  }

  return "neutral";
}

function visualizationVariant(type: VisualizationType): VisualizationViewModel["variant"] {
  switch (type) {
    case "class_balance":
    case "model_comparison":
    case "feature_importance":
      return "bars";
    case "correlation_heatmap":
    case "confusion_matrix":
      return "grid";
    case "roc_curve":
    case "pr_curve":
    case "residual_plot":
    case "actual_vs_predicted":
      return "line";
    case "experiment_graph":
      return "graph";
    default:
      return "fallback";
  }
}

function message(
  id: string,
  label: string,
  text: string,
  tone: WorkbenchMessage["tone"],
): WorkbenchMessage {
  return { id, label, text, tone };
}
