import { promises as fs } from "node:fs";
import { existsSync } from "node:fs";
import os from "node:os";
import path from "node:path";

const RUNTIME_ROOT = path.join(os.tmpdir(), "ml-labs-runtime-bundles");
const RUNTIME_METADATA_FILE = "runtime-metadata.json";
const RUNTIME_TTL_MS = 2 * 60 * 60 * 1000;
const MAX_RUNTIME_BUNDLES = 10;

type RuntimeMetadata = {
  createdAtEpochMs?: number;
};

export class RuntimeBundleExpiredError extends Error {
  constructor(runId: string) {
    super(`Run '${runId}' has expired. Re-run the lab to regenerate a prediction bundle.`);
    this.name = "RuntimeBundleExpiredError";
  }
}

export class RuntimeBundleMissingError extends Error {
  constructor(runId: string) {
    super(`Run '${runId}' was not found.`);
    this.name = "RuntimeBundleMissingError";
  }
}

export async function prepareRuntimeBundle(runId: string): Promise<string> {
  await ensureRuntimeRoot();
  await cleanupRuntimeBundles();

  const bundleDir = getRuntimeBundleDir(runId);
  await fs.rm(bundleDir, { recursive: true, force: true });
  await fs.mkdir(bundleDir, { recursive: true });

  return bundleDir;
}

export async function resolveRuntimeBundle(runId: string): Promise<string> {
  await ensureRuntimeRoot();
  await cleanupRuntimeBundles();

  const bundleDir = getRuntimeBundleDir(runId);
  if (!existsSync(bundleDir)) {
    throw new RuntimeBundleMissingError(runId);
  }

  const createdAtEpochMs = await readCreatedAt(bundleDir);
  if (Date.now() - createdAtEpochMs > RUNTIME_TTL_MS) {
    await fs.rm(bundleDir, { recursive: true, force: true });
    throw new RuntimeBundleExpiredError(runId);
  }

  return bundleDir;
}

export async function removeRuntimeBundle(runId: string): Promise<void> {
  await fs.rm(getRuntimeBundleDir(runId), { recursive: true, force: true });
}

function getRuntimeBundleDir(runId: string): string {
  return path.join(RUNTIME_ROOT, runId);
}

async function ensureRuntimeRoot(): Promise<void> {
  await fs.mkdir(RUNTIME_ROOT, { recursive: true });
}

async function cleanupRuntimeBundles(): Promise<void> {
  if (!existsSync(RUNTIME_ROOT)) {
    return;
  }

  const entries = await fs.readdir(RUNTIME_ROOT, { withFileTypes: true });
  const bundleInfos = await Promise.all(
    entries
      .filter((entry) => entry.isDirectory())
      .map(async (entry) => {
        const bundleDir = path.join(RUNTIME_ROOT, entry.name);
        const createdAtEpochMs = await readCreatedAt(bundleDir);
        return {
          bundleDir,
          createdAtEpochMs,
        };
      }),
  );

  const freshBundles = bundleInfos.filter(
    (bundleInfo) => Date.now() - bundleInfo.createdAtEpochMs <= RUNTIME_TTL_MS,
  );
  const expiredBundles = bundleInfos.filter(
    (bundleInfo) => Date.now() - bundleInfo.createdAtEpochMs > RUNTIME_TTL_MS,
  );

  await Promise.all(
    expiredBundles.map((bundleInfo) =>
      fs.rm(bundleInfo.bundleDir, { recursive: true, force: true }),
    ),
  );

  const overflowBundles = freshBundles
    .sort((left, right) => right.createdAtEpochMs - left.createdAtEpochMs)
    .slice(MAX_RUNTIME_BUNDLES);

  await Promise.all(
    overflowBundles.map((bundleInfo) =>
      fs.rm(bundleInfo.bundleDir, { recursive: true, force: true }),
    ),
  );
}

async function readCreatedAt(bundleDir: string): Promise<number> {
  const metadataPath = path.join(bundleDir, RUNTIME_METADATA_FILE);
  if (existsSync(metadataPath)) {
    try {
      const metadata = JSON.parse(
        await fs.readFile(metadataPath, "utf8"),
      ) as RuntimeMetadata;
      if (typeof metadata.createdAtEpochMs === "number") {
        return metadata.createdAtEpochMs;
      }
    } catch {
      // Fall back to filesystem timestamps below.
    }
  }

  const stats = await fs.stat(bundleDir);
  return stats.mtimeMs;
}
