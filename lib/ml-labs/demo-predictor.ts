import { getDemoLabRunResult, getDemoScenarioForRunId } from "@/lib/ml-labs/demo-result";
import type { LabPredictionResponse, ProblemType } from "@/lib/ml-labs/types";

const REGION_OFFSETS: Record<string, number> = {
  northeast: 1800,
  northwest: 1200,
  southeast: 2200,
  southwest: 1000,
};

const CONTRACT_OFFSETS: Record<string, number> = {
  monthly: 1.2,
  one_year: 0.2,
  two_year: -0.55,
};

const INTERNET_SERVICE_OFFSETS: Record<string, number> = {
  dsl: 0.1,
  fiber: 0.38,
  none: -0.28,
};

export function predictDemoRun(
  runId: string,
  input: Record<string, unknown>,
): LabPredictionResponse {
  const scenario = getDemoScenarioForRunId(runId);
  if (!scenario) {
    throw new Error(`Unknown demo run '${runId}'.`);
  }

  return scenario === "classification"
    ? predictClassificationDemo(runId, input)
    : predictRegressionDemo(runId, input);
}

function predictClassificationDemo(
  runId: string,
  input: Record<string, unknown>,
): LabPredictionResponse {
  const schema = getDemoLabRunResult("classification").predictionInputSchema;
  if (!schema) {
    throw new Error("Classification demo prediction schema is unavailable.");
  }

  const row = validateInput("classification", input);
  const contractType = String(row.contract_type);
  const internetService = String(row.internet_service);
  const autopay = String(row.autopay);

  const score =
    -1.75 +
    Number(row.monthly_charges) * 0.031 +
    Number(row.support_tickets) * 0.52 -
    Number(row.tenure_months) * 0.058 +
    (CONTRACT_OFFSETS[contractType] ?? 0) +
    (INTERNET_SERVICE_OFFSETS[internetService] ?? 0) +
    (autopay === "no" ? 0.52 : -0.22);

  const probability = 1 / (1 + Math.exp(-score));
  const prediction = probability >= 0.5 ? "yes" : "no";
  const topFactors = [
    `Monthly charges at ${row.monthly_charges} remain a strong churn driver in the demo model.`,
    `Support ticket volume (${row.support_tickets}) pushed the risk profile ${Number(row.support_tickets) >= 3 ? "up" : "down"}.`,
    `Contract type '${contractType}' materially influenced the final probability.`,
  ];

  return {
    runId,
    problemType: "classification",
    prediction,
    probability: round(probability),
    explanation: `The bundled churn demo model assigns ${Math.round(probability * 100)}% confidence to class '${prediction}'.`,
    topFactors,
  };
}

function predictRegressionDemo(
  runId: string,
  input: Record<string, unknown>,
): LabPredictionResponse {
  const schema = getDemoLabRunResult("regression").predictionInputSchema;
  if (!schema) {
    throw new Error("Regression demo prediction schema is unavailable.");
  }

  const row = validateInput("regression", input);
  const smokerLift = String(row.smoker) === "yes" ? 23800 : 0;
  const bmiLift = Math.max(Number(row.bmi) - 21, 0) * 320;
  const ageLift = Number(row.age) * 245;
  const childLift = Number(row.children) * 410;
  const prediction = Math.round(
    1600 + ageLift + bmiLift + childLift + smokerLift + (REGION_OFFSETS[String(row.region)] ?? 0),
  );

  return {
    runId,
    problemType: "regression",
    prediction,
    unit: "USD / year",
    explanation:
      "The bundled insurance demo regression model estimates annual charges from demographic and policy attributes.",
    topFactors: [
      String(row.smoker) === "yes"
        ? "Smoking status contributed the strongest positive lift."
        : "Non-smoker status kept the risk-adjusted estimate lower.",
      `BMI contributed approximately $${Math.round(bmiLift).toLocaleString()} to the estimate.`,
      `Age contributed approximately $${Math.round(ageLift).toLocaleString()} to the estimate.`,
    ],
  };
}

function validateInput(
  scenario: ProblemType,
  input: Record<string, unknown>,
): Record<string, string | number> {
  const schema = getDemoLabRunResult(scenario).predictionInputSchema;
  if (!schema) {
    throw new Error(`No demo input schema is available for scenario '${scenario}'.`);
  }

  const normalized: Record<string, string | number> = {};
  for (const field of schema.fields) {
    const rawValue = input[field.name];
    if (rawValue === undefined || rawValue === null || rawValue === "") {
      throw new Error(`Missing required field '${field.name}'.`);
    }

    if (field.kind === "number") {
      const numericValue = Number(rawValue);
      if (Number.isNaN(numericValue)) {
        throw new Error(`Field '${field.name}' must be numeric.`);
      }
      normalized[field.name] = numericValue;
      continue;
    }

    const stringValue = String(rawValue);
    if (field.options && field.options.length > 0 && !field.options.includes(stringValue)) {
      throw new Error(`Field '${field.name}' must be one of: ${field.options.join(", ")}.`);
    }
    normalized[field.name] = stringValue;
  }

  return normalized;
}

function round(value: number): number {
  return Math.round(value * 10000) / 10000;
}
