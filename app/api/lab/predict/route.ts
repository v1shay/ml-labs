import { NextResponse } from "next/server";
import { demoPredictionExamples, predictDemoInsuranceCharge } from "@/lib/ml-labs/demo-predictor";
import type { DemoPredictInput } from "@/lib/ml-labs/types";

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as Partial<DemoPredictInput>;
    const input = validatePredictInput(body);
    return NextResponse.json(predictDemoInsuranceCharge(input));
  } catch (error) {
    const details = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json(
      {
        error: "Invalid demo prediction payload.",
        details,
        examples: demoPredictionExamples,
      },
      { status: 400 },
    );
  }
}

function validatePredictInput(body: Partial<DemoPredictInput>): DemoPredictInput {
  const regions = new Set(["northeast", "northwest", "southeast", "southwest"]);
  const sexes = new Set(["female", "male"]);

  if (
    typeof body.age !== "number" ||
    typeof body.bmi !== "number" ||
    typeof body.children !== "number" ||
    typeof body.smoker !== "boolean" ||
    typeof body.sex !== "string" ||
    typeof body.region !== "string"
  ) {
    throw new Error("Expected age, sex, bmi, children, smoker, and region fields.");
  }

  if (!sexes.has(body.sex) || !regions.has(body.region)) {
    throw new Error("Sex or region value is outside the supported demo domain.");
  }

  return {
    age: body.age,
    sex: body.sex as DemoPredictInput["sex"],
    bmi: body.bmi,
    children: body.children,
    smoker: body.smoker,
    region: body.region as DemoPredictInput["region"],
  };
}

