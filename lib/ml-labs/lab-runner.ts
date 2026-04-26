import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { buildCompleteRun } from "@/lib/ml-labs/report-generator";
import type {
  AgentTraceItem,
  CriticReport,
  DatasetProfile,
  LabRunResult,
  LeaderboardEntry,
  PythonRunnerResult,
  Visualization,
} from "@/lib/ml-labs/types";

const execFileAsync = promisify(execFile);

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

  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "ml-labs-"));
  const tempFilePath = path.join(tempDir, sanitizeFilename(file.name));

  try {
    const fileBuffer = Buffer.from(await file.arrayBuffer());
    await fs.writeFile(tempFilePath, fileBuffer);

    const runnerResult = await executePythonRunner(tempFilePath, targetColumn, intentPrompt);
    return normalizeRunnerResult(runnerResult, intentPrompt);
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
}

async function executePythonRunner(
  csvPath: string,
  targetColumn: string,
  intentPrompt?: string,
): Promise<PythonRunnerResult> {
  const projectRoot = process.cwd();
  const pythonExecutable = process.env.ML_LABS_PYTHON || preferredPythonBinary(projectRoot);
  const scriptPath = path.join(projectRoot, "scripts", "ml_labs_runner.py");

  const args = [scriptPath, "--csv", csvPath, "--target", targetColumn];
  if (intentPrompt) {
    args.push("--intent", intentPrompt);
  }

  try {
    const { stdout } = await execFileAsync(pythonExecutable, args, {
      cwd: projectRoot,
      env: process.env,
      maxBuffer: 10 * 1024 * 1024,
    });

    return JSON.parse(stdout) as PythonRunnerResult;
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

function normalizeRunnerResult(
  runnerResult: PythonRunnerResult,
  intentPrompt?: string,
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
      runId: buildRunId(runnerResult.datasetProfile.targetColumn),
      intentPrompt,
      datasetProfile: runnerResult.datasetProfile,
      leaderboard,
      criticReport: runnerResult.criticReport,
    },
    {
      agentTrace,
      visualizations: normalizeVisualizations(runnerResult.visualizations),
    },
  );
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
      status: "complete",
      message: `Ingested ${datasetProfile.rows} rows across ${datasetProfile.columns} columns from the uploaded CSV.`,
    },
    {
      agent: "Schema Validation Agent",
      status: "complete",
      message: `Validated target column "${datasetProfile.targetColumn}" and inferred ${datasetProfile.problemType} behavior.`,
    },
    {
      agent: "Data Profiling Agent",
      status: "complete",
      message: `Separated ${datasetProfile.numericColumns.length} numeric and ${datasetProfile.categoricalColumns.length} categorical features.`,
    },
    {
      agent: "Missing Value Audit Agent",
      status: warningStatus,
      message:
        criticReport.warnings[0] ??
        "No major missing-value concerns were discovered in the primary training path.",
    },
    {
      agent: "Feature Planning Agent",
      status: "complete",
      message:
        "Prepared a preprocessing graph with numeric median imputation and categorical one-hot encoding.",
    },
    {
      agent: "Baseline Agent",
      status: "complete",
      message: `Benchmarked the baseline before training the model family sweep.`,
    },
  ];

  leaderboard
    .filter((entry) => entry.family.toLowerCase() !== "baseline")
    .forEach((entry) => {
      trace.push({
        agent: `${entry.family} Agent`,
        status: "complete",
        message: `${entry.modelName} finished with ${entry.metricName} ${entry.score.toFixed(3)} on held-out data.`,
      });
    });

  modelFailures.forEach((failure) => {
    trace.push({
      agent: "Fallback Modeling Agent",
      status: "warning",
      message: failure,
    });
  });

  trace.push(
    {
      agent: "Evaluation Agent",
      status: "complete",
      message: `${bestModel.modelName} won the leaderboard and now anchors the final summary.`,
    },
    {
      agent: "Critic Agent",
      status: warningStatus,
      message:
        criticReport.nextExperiments[0] ??
        "The critic found no severe blockers and produced a clean export package.",
    },
    {
      agent: "Report Agent",
      status: "complete",
      message: "Generated report markdown plus reusable code artifacts for export.",
    },
  );

  return trace;
}

function normalizeVisualizations(visualizations: Visualization[]): Visualization[] {
  return visualizations.map((visualization) => ({
    ...visualization,
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
    .find((line) => line.startsWith("ValueError:") || line.startsWith("RuntimeError:"));

  if (labeledLine) {
    return labeledLine.replace(/^[A-Za-z]+Error:\s*/, "");
  }

  return lines[lines.length - 1] ?? normalized;
}
