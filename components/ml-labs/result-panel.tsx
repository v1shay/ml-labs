import { VisualizationDeck } from "@/components/ml-labs/visualization-panel";
import type {
  ArtifactTab,
  DatasetMetricCard,
  VisualizationViewModel,
} from "@/lib/ml-labs/workbench-normalizer";
import type { LabRunResult, LeaderboardEntry } from "@/lib/ml-labs/types";

type ResultPanelProps = {
  result: LabRunResult | null;
  metricCards: DatasetMetricCard[];
  leaderboard: LeaderboardEntry[];
  criticBlocks: Array<{ title: string; items: string[] }>;
  artifactTabs: ArtifactTab[];
  selectedArtifactId: string | null;
  onArtifactSelect: (artifactId: string) => void;
  visualizations: VisualizationViewModel[];
  isVisible: boolean;
  isFocused: boolean;
};

export function ResultPanel({
  result,
  metricCards,
  leaderboard,
  criticBlocks,
  artifactTabs,
  selectedArtifactId,
  onArtifactSelect,
  visualizations,
  isVisible,
  isFocused,
}: ResultPanelProps) {
  const activeArtifact =
    artifactTabs.find((artifact) => artifact.id === selectedArtifactId) ?? artifactTabs[0] ?? null;

  return (
    <section
      className={[
        "workbench-panel",
        "result-panel",
        isVisible ? "panel-visible" : "panel-hidden",
        isFocused ? "panel-focused" : "panel-background",
      ].join(" ")}
    >
      <div className="panel-header">
        <div>
          <span className="panel-kicker">Panel 03</span>
          <h2>Structured dataset profile</h2>
        </div>
        <span className="status-pill status-complete">{result ? "complete" : "waiting"}</span>
      </div>

      {result ? (
        <div className="result-layout">
          <section className="result-section">
            <div className="section-heading">
              <span className="section-kicker">Dataset Profile</span>
              <h3>{result.datasetProfile.targetSummary}</h3>
            </div>

            <div className="metric-grid">
              {metricCards.map((card) => (
                <article key={card.label} className="metric-card">
                  <span>{card.label}</span>
                  <strong>{card.value}</strong>
                  <p>{card.hint}</p>
                </article>
              ))}
            </div>
          </section>

          <section className="result-section split">
            <div className="leaderboard-card">
              <div className="section-heading compact">
                <span className="section-kicker">Leaderboard</span>
                <h3>Model comparison</h3>
              </div>
              <div className="leaderboard-table">
                <div className="leaderboard-head">
                  <span>Model</span>
                  <span>Metric</span>
                  <span>Score</span>
                </div>
                {leaderboard.map((entry) => (
                  <div key={entry.modelName} className="leaderboard-row">
                    <div>
                      <strong>{entry.modelName}</strong>
                      <p>{entry.family}</p>
                    </div>
                    <span>{entry.metricName}</span>
                    <span>{entry.score.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="critic-grid">
              {criticBlocks.map((block) => (
                <article key={block.title} className="critic-card">
                  <span className="section-kicker">{block.title}</span>
                  {block.items.length ? (
                    block.items.map((item) => <p key={item}>{item}</p>)
                  ) : (
                    <p>No major notes surfaced in this category.</p>
                  )}
                </article>
              ))}
            </div>
          </section>

          <VisualizationDeck visualizations={visualizations} />

          <section className="result-section split">
            <article className="artifact-card">
              <div className="section-heading compact">
                <span className="section-kicker">Artifacts</span>
                <h3>Reusable exports</h3>
              </div>

              <div className="artifact-tabs">
                {artifactTabs.map((artifact) => (
                  <button
                    key={artifact.id}
                    type="button"
                    className={artifact.id === activeArtifact?.id ? "artifact-tab active" : "artifact-tab"}
                    onClick={() => onArtifactSelect(artifact.id)}
                  >
                    {artifact.label}
                  </button>
                ))}
              </div>

              <div className="artifact-content">
                <p>{activeArtifact?.content ?? "No artifact content available."}</p>
              </div>
            </article>

            <article className="report-card">
              <div className="section-heading compact">
                <span className="section-kicker">Report</span>
                <h3>Research layer</h3>
              </div>
              <div className="report-body">
                <p>{result.finalReportMarkdown}</p>
              </div>
            </article>
          </section>
        </div>
      ) : (
        <div className="result-empty">
          <p>The structured result panel will animate in after the first successful run.</p>
        </div>
      )}
    </section>
  );
}
