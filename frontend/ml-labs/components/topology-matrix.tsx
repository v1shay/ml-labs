"use client";

import { useEffect, useRef } from "react";
import cytoscape, { type Core, type ElementDefinition } from "cytoscape";
import { STAGES, type StageStatus } from "@/frontend/ml-labs/lib/stages";

type TopologyMatrixProps = {
  selectedStageId: string;
  stageStatusMap: Record<string, StageStatus>;
  onSelectStage: (stageId: string) => void;
};

const EDGES: Array<[string, string]> = [
  ["source-intake", "source-resolution"],
  ["source-resolution", "schema-profiling"],
  ["schema-profiling", "target-framing"],
  ["target-framing", "preprocessing"],
  ["preprocessing", "baseline"],
  ["baseline", "linear-model"],
  ["linear-model", "tree-model"],
  ["tree-model", "boosted-model"],
  ["boosted-model", "evaluation"],
  ["evaluation", "critic"],
  ["critic", "export"],
];

export function TopologyMatrix({
  selectedStageId,
  stageStatusMap,
  onSelectStage,
}: TopologyMatrixProps) {
  const graphRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);

  useEffect(() => {
    if (!graphRef.current) {
      return;
    }

    const cy = cytoscape({
      container: graphRef.current,
      elements: buildElements(stageStatusMap),
      layout: { name: "preset" },
      minZoom: 0.5,
      maxZoom: 1.8,
      wheelSensitivity: 0.12,
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            shape: "round-rectangle",
            width: 168,
            height: 56,
            color: "#efe8df",
            "font-family": "var(--font-mono-custom)",
            "font-size": 10,
            "text-transform": "uppercase",
            "text-wrap": "wrap",
            "text-max-width": "118px",
            "text-valign": "center",
            "background-color": "#1a202a",
            "border-width": 1.2,
            "border-color": "rgba(136,154,178,0.38)",
            "text-outline-opacity": 0,
          },
        },
        {
          selector: "node.active",
          style: {
            "border-color": "#85b7eb",
            "border-width": 2.5,
            "overlay-color": "rgba(55,138,221,0.26)",
            "overlay-opacity": 0.22,
          },
        },
        {
          selector: ".complete",
          style: {
            "background-color": "#17202b",
            "border-color": "#1d9e75",
          },
        },
        {
          selector: ".running",
          style: {
            "background-color": "#243347",
            "border-color": "#85b7eb",
          },
        },
        {
          selector: ".queued",
          style: {
            "background-color": "#151c26",
            "border-color": "#889ab2",
            "border-style": "dashed",
          },
        },
        {
          selector: ".warning",
          style: {
            "background-color": "#2a2118",
            "border-color": "#ef9f27",
          },
        },
        {
          selector: ".failed",
          style: {
            "background-color": "#2d1717",
            "border-color": "#e24b4a",
          },
        },
        {
          selector: "edge",
          style: {
            width: 1.5,
            "line-color": "rgba(136,154,178,0.24)",
            "target-arrow-shape": "triangle",
            "target-arrow-color": "rgba(136,154,178,0.24)",
            "curve-style": "bezier",
          },
        },
        {
          selector: "edge.flow-complete",
          style: {
            width: 2.5,
            "line-color": "#378add",
            "target-arrow-color": "#378add",
          },
        },
      ],
    });

    cy.on("tap", "node", (event) => {
      const nodeId = event.target.id();
      onSelectStage(nodeId);
    });

    cyRef.current = cy;
    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, [onSelectStage, stageStatusMap]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) {
      return;
    }

    cy.nodes().removeClass("active");
    const activeNode = cy.getElementById(selectedStageId);
    if (activeNode.nonempty()) {
      activeNode.addClass("active");
    }
  }, [selectedStageId]);

  return (
    <div className="topology-matrix-shell">
      <div className="topology-toolbar">
        <span className="shell-kicker">ML Topology Matrix</span>
        <div className="topology-legend">
          <span>Resolved</span>
          <span>Queued</span>
          <span>Warning</span>
        </div>
      </div>
      <div ref={graphRef} className="topology-canvas" />
    </div>
  );
}

function buildElements(stageStatusMap: Record<string, StageStatus>): ElementDefinition[] {
  const nodes = STAGES.map((stage) => ({
    data: { id: stage.id, label: stage.shortLabel },
    position: {
      x: 140 + stage.x * 220,
      y: 90 + stage.y * 165,
    },
    classes: stageStatusMap[stage.id],
  }));

  const edges = EDGES.map(([source, target]) => ({
    data: {
      id: `${source}-${target}`,
      source,
      target,
    },
    classes:
      stageStatusMap[source] === "complete" || stageStatusMap[source] === "warning"
        ? "flow-complete"
        : "",
  }));

  return [...nodes, ...edges];
}
