import { promises as fs } from "node:fs";
import { existsSync } from "node:fs";
import { execFile } from "node:child_process";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { buildCompleteRun } from "@/lib/ml-labs/report-generator";
import {
  prepareRuntimeBundle,
  removeRuntimeBundle,
  resolveRuntimeBundle,
  RuntimeBundleExpiredError,
  RuntimeBundleMissingError,
} from "@/lib/ml-labs/runtime-store";
import type {
  AgentTraceItem,
  CriticReport,
  DatasetProfile,
  LabPredictionResponse,
  LabRunResult,
  LeaderboardEntry,
  ProblemType,
  PythonRunnerResult,
  Visualization,
} from "@/lib/ml-labs/types";

const execFileAsync = promisify(execFile);

type BuildRunOptions = {
  runId: string;
  scenario?: ProblemType;
  intentPrompt?: string;
};

export async function runLab({
  file,
  targetColumn,
  intentPrompt,
}: {
  file: File;
  targetColumn: string;
  intentPrompt?: string;
}): Promise<LabRunResult> {
  if (!targetColumn.trim()) {
    throw new Error("A target column is required.");
  }

  if (!file.name.toLowerCase().endsWith(".csv")) {
    throw new Error("Only CSV uploads are supported in this MVP.");
  }

  const runId = buildRunId(targetColumn);
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "ml-labs-"));
  const tempFilePath = path.join(tempDir, sanitizeFilename(file.name));
  const bundleDir = await prepareRuntimeBundle(runId);

  try {
    const fileBuffer = Buffer.from(await file.arrayBuffer());
    await fs.writeFile(tempFilePath, fileBuffer);

    const runnerResult = await executePythonTrain({
      bundleDir,
      csvPath: tempFilePath,
      intentPrompt,
      runId,
      targetColumn,
    });

    return buildLabRunFromRunnerResult(runnerResult, {
      runId,
      scenario: runnerResult.datasetProfile.problemType,
      intentPrompt,
    });
  } catch (error) {
    await removeRuntimeBundle(runId);
    throw error;
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
}

export async function predictLabRun({
  input,
  runId,
}: {
  input: Record<string, unknown>;
  runId: string;
}): Promise<LabPredictionResponse> {
  const bundleDir = await resolveRuntimeBundle(runId);
  return executePythonPredict({ bundleDir, input, runId });
}

export function buildLabRunFromRunnerResult(
  runnerResult: PythonRunnerResult,
  options: BuildRunOptions,
): LabRunResult {
  const leaderboard = normalizeLeaderboard(runnerResult.leaderboard);
  const agentTrace = buildAgentTrace(
    runnerResult.datasetProfile,
    leaderboard,
    runnerResult.criticReport,
    runnerResult.metadata?.modelFailures ?? [],
  );

  return buildCompleteRun(
    {
      runId: options.runId,
      scenario: options.scenario ?? runnerResult.datasetProfile.problemType,
      intentPrompt: options.intentPrompt,
      datasetProfile: runnerResult.datasetProfile,
      leaderboard,
      criticReport: runnerResult.criticReport,
      predictionInputSchema: runnerResult.predictionInputSchema,
    },
    {
      agentTrace,
      visualizations: normalizeVisualizations(runnerResult.visualizations),
    },
  );
}

export { RuntimeBundleExpiredError, RuntimeBundleMissingError };

async function executePythonTrain({
  bundleDir,
  csvPath,
  intentPrompt,
  runId,
  targetColumn,
}: {
  bundleDir: string;
  csvPath: string;
  intentPrompt?: string;
  runId: string;
  targetColumn: string;
}): Promise<PythonRunnerResult> {
  const args = [
    "train",
    "--csv",
    csvPath,
    "--target",
    targetColumn,
    "--run-id",
    runId,
    "--bundle-dir",
    bundleDir,
  ];

  if (intentPrompt) {
    args.push("--intent", intentPrompt);
  }

  return executePythonCommand<PythonRunnerResult>(args);
}

async function executePythonPredict({
  bundleDir,
  input,
  runId,
}: {
  bundleDir: string;
  input: Record<string, unknown>;
  runId: string;
}): Promise<LabPredictionResponse> {
  const response = await executePythonCommand<LabPredictionResponse>([
    "predict",
    "--bundle-dir",
    bundleDir,
    "--run-id",
    runId,
    "--input-json",
    JSON.stringify(input),
  ]);

  return response;
}

async function executePythonCommand<T>(args: string[]): Promise<T> {
  const projectRoot = process.cwd();
  const pythonExecutable = process.env.ML_LABS_PYTHON || preferredPythonBinary(projectRoot);
  const scriptPath = path.join(projectRoot, "scripts", "ml_labs_runner.py");

  try {
    const { stdout } = await execFileAsync(pythonExecutable, [scriptPath, ...args], {
      cwd: projectRoot,
      env: process.env,
      maxBuffer: 10 * 1024 * 1024,
    });

    return JSON.parse(stdout) as T;
  } catch (error) {
    const stderr =
      typeof error === "object" && error !== null && "stderr" in error
        ? String(error.stderr ?? "")
        : "";
    const cleanedStderr = extractPythonFailure(stderr);
    const message =
      cleanedStderr ||
      (error instanceof Error ? error.message : "The Python experiment engine failed unexpectedly.");
    throw new Error(`ML-Labs Python runner failed: ${message}`);
  }
}

function normalizeLeaderboard(entries: LeaderboardEntry[]): LeaderboardEntry[] {
  const sorted = [...entries].sort((left, right) => right.score - left.score);
  const baseline =
    sorted.find((entry) => entry.family.toLowerCase() === "baseline") ?? sorted[sorted.length - 1];

  return sorted.map((entry) => ({
    ...entry,
    score: round(entry.score),
    trainScore: entry.trainScore !== undefined ? round(entry.trainScore) : undefined,
    testScore: entry.testScore !== undefined ? round(entry.testScore) : undefined,
    improvementOverBaseline: round(entry.score - baseline.score),
  }));
}

function buildAgentTrace(
  datasetProfile: DatasetProfile,
  leaderboard: LeaderboardEntry[],
  criticReport: CriticReport,
  modelFailures: string[],
): AgentTraceItem[] {
  const bestModel = leaderboard[0];
  const warningStatus = criticReport.warnings.length > 0 ? "warning" : "complete";

  const trace: AgentTraceItem[] = [
    {
      agent: "Data Intake Agent",
      stageId: "intake",
      status: "complete",
      message: `Ingested ${datasetProfile.rows} rows across ${datasetProfile.columns} columns from the uploaded CSV.`,
    },
    {
      agent: "Schema Validation Agent",
      stageId: "schema-validation",
      status: "complete",
      message: `Validated target column "${datasetProfile.targetColumn}" and inferred ${datasetProfile.problemType} behavior.`,
    },
    {
      agent: "Data Profiling Agent",
      stageId: "profiling",
      status: "complete",
      message: `Separated ${datasetProfile.numericColumns.length} numeric and ${datasetProfile.categoricalColumns.length} categorical features.`,
    },
    {
      agent: "Problem Framing Agent",
      stageId: "framing",
      status: "complete",
      message: `Selected ${metricDescription(datasetProfile.problemType)} as the primary evaluation frame for the lab run.`,
    },
    {
      agent: "Feature Planning Agent",
      stageId: "preprocessing",
      status: "complete",
      message:
        "Prepared numeric median imputation and categorical one-hot encoding inside a consistent train-test pipeline.",
    },
    {
      agent: "Baseline Agent",
      stageId: "baseline",
      status: "complete",
      message: "Benchmarked the baseline before the broader model family sweep.",
    },
  ];

  leaderboard
    .filter((entry) => entry.family.toLowerCase() !== "baseline")
    .forEach((entry) => {
      trace.push({
        agent: `${entry.family} Agent`,
        stageId: familyToStageId(entry.family),
        status: "complete",
        message: `${entry.modelName} finished with ${entry.metricName} ${entry.score.toFixed(3)} on held-out data.`,
      });
    });

  modelFailures.forEach((failure) => {
    trace.push({
      agent: "Fallback Modeling Agent",
      stageId: "fallback",
      status: "warning",
      message: failure,
    });
  });

  trace.push(
    {
      agent: "Evaluation Agent",
      stageId: "evaluation",
      status: "complete",
      message: `${bestModel.modelName} won the leaderboard and anchors the final summary.`,
    },
    {
      agent: "Critic Agent",
      stageId: "critic",
      status: warningStatus,
      message:
        criticReport.nextExperiments[0] ??
        "The critic found no severe blockers and produced a clean export package.",
    },
    {
      agent: "Report Agent",
      stageId: "packaging",
      status: "complete",
      message: "Generated report markdown plus runnable code artifacts for export.",
    },
  );

  return trace;
}

function normalizeVisualizations(visualizations: Visualization[]): Visualization[] {
  return visualizations.map((visualization, index) => ({
    ...visualization,
    id: visualization.id || `viz-${index + 1}`,
    data: visualization.data,
  }));
}

function preferredPythonBinary(projectRoot: string): string {
  const localPython = path.join(projectRoot, ".venv", "bin", "python");
  return existsSync(localPython) ? localPython : "python3";
}

function buildRunId(targetColumn: string): string {
  return `lab-${targetColumn.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-${Date.now()}`;
}

function sanitizeFilename(filename: string): string {
  return filename.replace(/[^a-zA-Z0-9._-]/g, "_");
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function familyToStageId(family: string): string {
  const normalized = family.toLowerCase();
  if (normalized.includes("linear")) {
    return "linear-model";
  }

  if (normalized.includes("boost")) {
    return "boosted-trees";
  }

  if (normalized.includes("tree") || normalized.includes("forest")) {
    return "tree-model";
  }

  return "modeling";
}

function metricDescription(problemType: ProblemType): string {
  return problemType === "classification"
    ? "classification accuracy and ranking diagnostics"
    : "regression fit and residual diagnostics";
}

function extractPythonFailure(stderr: string): string {
  const normalized = stderr.trim();
  if (!normalized) {
    return "";
  }

  const lines = normalized
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const labeledLine = [...lines]
    .reverse()
    .find((line) => /^(Key|Runtime|Type|Value)Error:/.test(line));

  if (labeledLine) {
    return labeledLine.replace(/^[A-Za-z]+Error:\s*/, "");
  }

  return lines[lines.length - 1] ?? normalized;
}
