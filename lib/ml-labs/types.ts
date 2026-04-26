export type ProblemType = "classification" | "regression";
export type AgentStatus = "queued" | "running" | "complete" | "warning" | "failed";
export type VisualizationType =
  | "actual_vs_predicted"
  | "class_balance"
  | "correlation_heatmap"
  | "feature_importance"
  | "confusion_matrix"
  | "experiment_graph"
  | "missingness_summary"
  | "model_comparison"
  | "pr_curve"
  | "residual_plot"
  | "roc_curve";

export type PredictionInputKind = "boolean" | "number" | "string";

export type DatasetProfile = {
  rows: number;
  columns: number;
  targetColumn: string;
  problemType: ProblemType;
  numericColumns: string[];
  categoricalColumns: string[];
  missingValues: Record<string, number>;
  targetSummary: string;
};

export type AgentTraceItem = {
  agent: string;
  stageId: string;
  status: AgentStatus;
  message: string;
};

export type LeaderboardEntry = {
  modelName: string;
  family: string;
  metricName: string;
  score: number;
  trainScore?: number;
  testScore?: number;
  improvementOverBaseline?: number;
  notes?: string;
};

export type BestModelSummary = {
  modelName: string;
  metricName: string;
  score: number;
  baselineScore: number;
  absoluteImprovement: number;
  relativeImprovement: number;
  whyItWon: string;
};

export type CriticReport = {
  warnings: string[];
  failureModes: string[];
  nextExperiments: string[];
  limitations: string[];
};

export type Visualization = {
  id: string;
  stageId?: string;
  type: VisualizationType;
  title: string;
  data: unknown;
};

export type ArtifactType = "code" | "report" | "model";

export type LabArtifact = {
  filename: string;
  type: ArtifactType;
  content?: string;
  downloadUrl?: string;
};

export type LabRunResult = {
  runId: string;
  scenario?: ProblemType;
  datasetProfile: DatasetProfile;
  agentTrace: AgentTraceItem[];
  leaderboard: LeaderboardEntry[];
  bestModel: BestModelSummary;
  criticReport: CriticReport;
  visualizations: Visualization[];
  predictionInputSchema?: PredictionInputSchema;
  artifacts: LabArtifact[];
  finalReportMarkdown: string;
};

export type LabRunError = {
  error: string;
  details?: string;
};

export type PredictionInputField = {
  name: string;
  label: string;
  kind: PredictionInputKind;
  required: boolean;
  options?: string[];
  example?: boolean | number | string;
  description?: string;
};

export type PredictionInputSchema = {
  targetColumn: string;
  problemType: ProblemType;
  fields: PredictionInputField[];
};

export type LabPredictionRequest = {
  runId: string;
  input: Record<string, unknown>;
};

export type LabPredictionResponse = {
  runId: string;
  problemType: ProblemType;
  prediction: boolean | number | string;
  probability?: number;
  unit?: string;
  explanation: string;
  topFactors: string[];
};

export type PythonRunnerResult = {
  datasetProfile: DatasetProfile;
  leaderboard: LeaderboardEntry[];
  criticReport: CriticReport;
  visualizations: Visualization[];
  predictionInputSchema: PredictionInputSchema;
  metadata?: {
    targetMean?: number | null;
    targetStd?: number | null;
    modelFailures?: string[];
    intentPrompt?: string;
    trainingNote?: string | null;
  };
};
