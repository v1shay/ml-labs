export type ProblemType = "classification" | "regression";
export type AgentStatus = "queued" | "running" | "complete" | "warning" | "failed";
export type VisualizationType =
  | "feature_importance"
  | "confusion_matrix"
  | "residual_plot"
  | "experiment_graph";

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
  datasetProfile: DatasetProfile;
  agentTrace: AgentTraceItem[];
  leaderboard: LeaderboardEntry[];
  bestModel: BestModelSummary;
  criticReport: CriticReport;
  visualizations: Visualization[];
  artifacts: LabArtifact[];
  finalReportMarkdown: string;
};

export type LabRunError = {
  error: string;
  details?: string;
};

export type DemoPredictInput = {
  age: number;
  sex: "female" | "male";
  bmi: number;
  children: number;
  smoker: boolean;
  region: "northeast" | "northwest" | "southeast" | "southwest";
};

export type DemoPredictResponse = {
  prediction: number;
  unit: string;
  explanation: string;
  topFactors: string[];
};

export type PythonRunnerResult = {
  datasetProfile: DatasetProfile;
  leaderboard: LeaderboardEntry[];
  criticReport: CriticReport;
  visualizations: Visualization[];
  metadata?: {
    targetMean?: number | null;
    targetStd?: number | null;
    modelFailures?: string[];
    intentPrompt?: string;
  };
};
