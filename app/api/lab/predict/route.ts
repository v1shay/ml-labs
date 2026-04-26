import { NextResponse } from "next/server";
import { predictDemoRun } from "@/lib/ml-labs/demo-predictor";
import { getDemoLabRunResult, isDemoRunId } from "@/lib/ml-labs/demo-result";
import {
  predictLabRun,
  RuntimeBundleExpiredError,
  RuntimeBundleMissingError,
} from "@/lib/ml-labs/lab-runner";
import type { LabPredictionRequest, LabRunError } from "@/lib/ml-labs/types";

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as Partial<LabPredictionRequest>;
    if (typeof body.runId !== "string" || body.runId.trim().length === 0) {
      return NextResponse.json<LabRunError>(
        {
          error: "A non-empty `runId` field is required.",
        },
        { status: 400 },
      );
    }

    if (!body.input || typeof body.input !== "object" || Array.isArray(body.input)) {
      return NextResponse.json<LabRunError>(
        {
          error: "A JSON object is required under the `input` field.",
        },
        { status: 400 },
      );
    }

    if (isDemoRunId(body.runId)) {
      return NextResponse.json(predictDemoRun(body.runId, body.input as Record<string, unknown>));
    }

    const prediction = await predictLabRun({
      runId: body.runId,
      input: body.input as Record<string, unknown>,
    });

    return NextResponse.json(prediction);
  } catch (error) {
    if (error instanceof RuntimeBundleMissingError) {
      return NextResponse.json<LabRunError>(
        {
          error: "Unknown run ID.",
          details: error.message,
        },
        { status: 404 },
      );
    }

    if (error instanceof RuntimeBundleExpiredError) {
      return NextResponse.json<LabRunError>(
        {
          error: "Run expired.",
          details: error.message,
        },
        { status: 410 },
      );
    }

    const details = error instanceof Error ? error.message : "Unknown error";
    const demoSchema = getDemoLabRunResult("classification").predictionInputSchema;

    return NextResponse.json(
      {
        error: "Prediction request failed.",
        details,
        predictionInputSchema: demoSchema,
      },
      { status: 400 },
    );
  }
}
