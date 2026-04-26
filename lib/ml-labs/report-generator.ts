import type {
  BestModelSummary,
  CriticReport,
  DatasetProfile,
  LabArtifact,
  LabRunResult,
  LeaderboardEntry,
  ProblemType,
} from "@/lib/ml-labs/types";

const round = (value: number) => Math.round(value * 1000) / 1000;

type ReportBuildInput = {
  runId: string;
  intentPrompt?: string;
  datasetProfile: DatasetProfile;
  leaderboard: LeaderboardEntry[];
  criticReport: CriticReport;
};

export function buildBestModelSummary(
  datasetProfile: DatasetProfile,
  leaderboard: LeaderboardEntry[],
): BestModelSummary {
  const sorted = [...leaderboard].sort((a, b) => b.score - a.score);
  const baseline =
    sorted.find((entry) => entry.family.toLowerCase() === "baseline") ?? sorted[sorted.length - 1];
  const winner = sorted[0];
  const absoluteImprovement = round(winner.score - baseline.score);
  const relativeImprovement = round(
    (absoluteImprovement / Math.max(Math.abs(baseline.score), 0.000001)) * 100,
  );

  return {
    modelName: winner.modelName,
    metricName: winner.metricName,
    score: winner.score,
    baselineScore: baseline.score,
    absoluteImprovement,
    relativeImprovement,
    whyItWon: buildWhyItWon(datasetProfile.problemType, winner, baseline),
  };
}

export function buildArtifacts(
  input: ReportBuildInput,
  bestModel: BestModelSummary,
  reportMarkdown: string,
): LabArtifact[] {
  return [
    {
      filename: "train.py",
      type: "code",
      content: buildTrainArtifact(input.datasetProfile, bestModel),
    },
    {
      filename: "evaluate.py",
      type: "code",
      content: buildEvaluateArtifact(input.datasetProfile, bestModel),
    },
    {
      filename: "report.md",
      type: "report",
      content: reportMarkdown,
    },
  ];
}

export function buildFinalReportMarkdown({
  runId,
  intentPrompt,
  datasetProfile,
  leaderboard,
  criticReport,
}: ReportBuildInput): string {
  const bestModel = buildBestModelSummary(datasetProfile, leaderboard);
  const leaderboardTable = leaderboard
    .map(
      (entry) =>
        `| ${entry.modelName} | ${entry.family} | ${entry.metricName} | ${entry.score.toFixed(3)} | ${entry.improvementOverBaseline?.toFixed(3) ?? "0.000"} |`,
    )
    .join("\n");

  const intentLine = intentPrompt
    ? `Research intent: ${intentPrompt}`
    : "Research intent: Build the strongest reliable baseline-to-production tabular model for this dataset.";

  return `# ML-Labs Research Report

Run ID: ${runId}

${intentLine}

## Dataset Profile

- Rows: ${datasetProfile.rows}
- Columns: ${datasetProfile.columns}
- Target column: ${datasetProfile.targetColumn}
- Problem type: ${datasetProfile.problemType}
- Numeric columns: ${datasetProfile.numericColumns.join(", ") || "None"}
- Categorical columns: ${datasetProfile.categoricalColumns.join(", ") || "None"}
- Target summary: ${datasetProfile.targetSummary}

## Model Leaderboard

| Model | Family | Metric | Score | Improvement vs Baseline |
| --- | --- | --- | --- | --- |
${leaderboardTable}

## Best Model

- Winner: ${bestModel.modelName}
- Score: ${bestModel.score.toFixed(3)} ${bestModel.metricName}
- Baseline score: ${bestModel.baselineScore.toFixed(3)}
- Absolute improvement: ${bestModel.absoluteImprovement.toFixed(3)}
- Relative improvement: ${bestModel.relativeImprovement.toFixed(2)}%
- Why it won: ${bestModel.whyItWon}

## Critic Report

### Warnings
${renderBulletSection(criticReport.warnings)}

### Failure Modes
${renderBulletSection(criticReport.failureModes)}

### Next Experiments
${renderBulletSection(criticReport.nextExperiments)}

### Limitations
${renderBulletSection(criticReport.limitations)}

## Export Notes

- Generated artifacts include training, evaluation, and report templates.
- The frontend can expose these artifacts for copy/download without any extra backend work.
`;
}

export function buildCompleteRun(
  input: ReportBuildInput,
  overrides: Pick<LabRunResult, "agentTrace" | "visualizations">,
): LabRunResult {
  const bestModel = buildBestModelSummary(input.datasetProfile, input.leaderboard);
  const finalReportMarkdown = buildFinalReportMarkdown(input);

  return {
    runId: input.runId,
    datasetProfile: input.datasetProfile,
    agentTrace: overrides.agentTrace,
    leaderboard: input.leaderboard,
    bestModel,
    criticReport: input.criticReport,
    visualizations: overrides.visualizations,
    artifacts: buildArtifacts(input, bestModel, finalReportMarkdown),
    finalReportMarkdown,
  };
}

function buildWhyItWon(
  problemType: ProblemType,
  winner: LeaderboardEntry,
  baseline: LeaderboardEntry,
): string {
  const gap = winner.trainScore !== undefined && winner.testScore !== undefined
    ? round(winner.trainScore - winner.testScore)
    : null;

  const generalizationText =
    gap !== null
      ? `It held a train-test gap of ${gap.toFixed(3)}, which kept generalization acceptable for a hackathon MVP.`
      : "Its test-time behavior was stable enough to trust for the MVP demo.";

  const metricContext =
    problemType === "regression"
      ? `It improved the regression score over the baseline by ${round(winner.score - baseline.score).toFixed(3)}.`
      : `It raised classification performance above the baseline by ${round(winner.score - baseline.score).toFixed(3)}.`;

  return `${metricContext} ${generalizationText}`;
}

function buildTrainArtifact(datasetProfile: DatasetProfile, bestModel: BestModelSummary): string {
  return `import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Generated by ML-Labs for target: ${datasetProfile.targetColumn}
# Best model from the winning run: ${bestModel.modelName}

df = pd.read_csv("dataset.csv")
target = "${datasetProfile.targetColumn}"
X = df.drop(columns=[target])
y = df[target]

numeric_features = ${JSON.stringify(datasetProfile.numericColumns, null, 2)}
categorical_features = ${JSON.stringify(datasetProfile.categoricalColumns, null, 2)}

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

print("Attach the ${bestModel.modelName} estimator here and fit on X/y.")
`;
}

function buildEvaluateArtifact(datasetProfile: DatasetProfile, bestModel: BestModelSummary): string {
  return `import pandas as pd
from sklearn.metrics import ${datasetProfile.problemType === "regression" ? "r2_score" : "accuracy_score"}

# Generated by ML-Labs for ${bestModel.modelName}

df = pd.read_csv("dataset.csv")
target = "${datasetProfile.targetColumn}"

print("Load your trained pipeline, score it on held-out data, and compare against the baseline.")
`;
}

function renderBulletSection(lines: string[]): string {
  if (lines.length === 0) {
    return "- None";
  }

  return lines.map((line) => `- ${line}`).join("\n");
}

