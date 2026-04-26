import { NextResponse } from "next/server";
import { runLab } from "@/lib/ml-labs/lab-runner";
import type { LabRunError } from "@/lib/ml-labs/types";

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get("file");
    const targetColumn = formData.get("targetColumn");
    const intentPrompt = formData.get("intentPrompt");

    if (!(file instanceof File)) {
      return NextResponse.json<LabRunError>(
        { error: "A CSV file must be provided under the `file` field." },
        { status: 400 },
      );
    }

    if (typeof targetColumn !== "string" || targetColumn.trim().length === 0) {
      return NextResponse.json<LabRunError>(
        { error: "A non-empty `targetColumn` field is required." },
        { status: 400 },
      );
    }

    const result = await runLab({
      file,
      targetColumn,
      intentPrompt: typeof intentPrompt === "string" ? intentPrompt : undefined,
    });

    return NextResponse.json(result);
  } catch (error) {
    const details = error instanceof Error ? error.message : "Unknown error";
    const normalizedDetails = details.toLowerCase();
    const status =
      normalizedDetails.includes("only csv") ||
      normalizedDetails.includes("target column") ||
      normalizedDetails.includes("must contain at least one feature") ||
      normalizedDetails.includes("contains only missing values")
        ? 400
        : 500;

    return NextResponse.json<LabRunError>(
      {
        error: "ML-Labs could not complete the requested run.",
        details,
      },
      { status },
    );
  }
}
