import { NextResponse } from "next/server";
import { getDemoLabRunResult } from "@/lib/ml-labs/demo-result";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const intentPrompt = searchParams.get("intentPrompt") ?? undefined;

  return NextResponse.json(getDemoLabRunResult(intentPrompt));
}

