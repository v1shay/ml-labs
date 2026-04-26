import { NextResponse } from "next/server";
import type { LabRunError, ProblemType } from "@/lib/ml-labs/types";
import { getDemoLabRunResult } from "@/lib/ml-labs/demo-result";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const intentPrompt = searchParams.get("intentPrompt") ?? undefined;
  const scenario = (searchParams.get("scenario") ?? "classification") as ProblemType;

  if (scenario !== "classification" && scenario !== "regression") {
    return NextResponse.json<LabRunError>(
      {
        error: "Invalid demo scenario.",
        details: "Supported scenarios are `classification` and `regression`.",
      },
      { status: 400 },
    );
  }

  return NextResponse.json(getDemoLabRunResult(scenario, intentPrompt));
}
