import type { AgentLayer, AgentRole } from "./types";

// ── Agent definitions for TinyMetatron ───────────────────────────────────────

export interface AgentDef {
  callsign: string;
  name: string;
  layer: AgentLayer;
  role: string;
  summary: string;
  duties: string[];
}

const TMT_AGENTS: Record<AgentRole, AgentDef> = {
  validator: {
    callsign: "VALIDATOR",
    name: "Gate Enforcer",
    layer: "processing",
    role: "Runs generalize_loop.run_gate()",
    summary: "Gate-keeping agent that validates state transitions and enforces loop invariants before execution proceeds.",
    duties: [
      "Evaluate pre-conditions for all loop entries",
      "Validate state transition合法性",
      "Enforce phi-score thresholds before advancing",
      "Reject invalid state mutations",
      "Report gate violations to auditor",
    ],
  },
  synthesizer: {
    callsign: "SYNTHESIZER",
    name: "Zadkiel",
    layer: "integration",
    role: "Aggregates upstream outputs",
    summary: "Integrates and synthesizes outputs from multiple agents into coherent composite states for downstream consumption.",
    duties: [
      "Aggregate results from input and processing layers",
      "Resolve conflicts between agent outputs",
      "Produce unified state representations",
      "Distribute synthesized results to output layer",
      "Maintain coherence across integration boundaries",
    ],
  },
  workflow: {
    callsign: "WORKFLOW",
    name: "Loop Orchestrator",
    layer: "processing",
    role: "Runs train/corpus loops",
    summary: "Orchestrates the primary training and corpus processing loops, coordinating multi-step workflows across agents.",
    duties: [
      "Schedule loop iterations across agents",
      "Manage train step coordination",
      "Execute corpus processing pipelines",
      "Track loop progress and completion",
      "Handle loop-level error recovery",
    ],
  },
  observer: {
    callsign: "OBSERVER",
    name: "Raphael",
    layer: "input",
    role: "Telemetry collection",
    summary: "Collects and relays real-time telemetry data from all agents, monitoring system health and performance metrics.",
    duties: [
      "Aggregate metrics from all agents",
      "Monitor phi-alignment and resonance",
      "Track task completion rates",
      "Detect anomalies in agent behavior",
      "Stream telemetry via SSE to frontend",
    ],
  },
  archivist: {
    callsign: "ARCHIVIST",
    name: "Raziel",
    layer: "output",
    role: "Checkpoint registry",
    summary: "Maintains a persistent registry of checkpoint snapshots and state archives for recovery and replay.",
    duties: [
      "Save periodic state checkpoints",
      "Index checkpoint metadata",
      "Serve checkpoint restore requests",
      "Prune old checkpoints per retention policy",
      "Maintain checkpoint integrity checksums",
    ],
  },
  auditor: {
    callsign: "AUDITOR",
    name: "Cassiel",
    layer: "output",
    role: "Loop invariant validation",
    summary: "Validates loop invariants and audits execution traces to ensure protocol compliance and correctness.",
    duties: [
      "Verify loop invariant satisfaction",
      "Audit execution traces for policy violations",
      "Generate compliance reports",
      "Flag deviations from expected behavior",
      "Archive audit logs for forensics",
    ],
  },
  bronze: {
    callsign: "BRONZE",
    name: "Hesed",
    layer: "processing",
    role: "Safety + gate enforcement",
    summary: "Implements safety guardrails and bronze-tier gate enforcement for system protection.",
    duties: [
      "Enforce safety boundaries on all operations",
      "Validate resource allocation limits",
      "Block unsafe state transitions",
      "Monitor memory and compute budgets",
      "Trigger protective halts when needed",
    ],
  },
  federation: {
    callsign: "FEDERATION",
    name: "Camael",
    layer: "processing",
    role: "Multi-loop coordination",
    summary: "Coordinates execution across multiple parallel loops, managing cross-loop dependencies and synchronization.",
    duties: [
      "Synchronize parallel loop executions",
      "Resolve inter-loop dependencies",
      "Manage shared resource pools",
      "Coordinate cross-loop messaging",
      "Balance load across federated loops",
    ],
  },
  strategic: {
    callsign: "STRATEGIC",
    name: "Michael",
    layer: "processing",
    role: "Experiment path planning",
    summary: "Plans strategic experiment paths and optimizes the search over hyperparameter configurations.",
    duties: [
      "Generate experiment trajectories",
      "Evaluate exploration vs exploitation tradeoffs",
      "Plan multi-step experiment sequences",
      "Adapt strategy based on results",
      "Optimize for overall phi-score improvement",
    ],
  },
  bitnet: {
    callsign: "BITNET",
    name: "Gabriel",
    layer: "processing",
    role: "Entropy/quality scoring",
    summary: "Computes entropy and quality scores for states, guiding selection toward high-fidelity configurations.",
    duties: [
      "Calculate entropy of agent states",
      "Score state quality and diversity",
      "Filter low-quality candidates",
      "Report entropy trends to observer",
      "Guide sampling toward high-signal states",
    ],
  },
  harmonic: {
    callsign: "HARMONIC",
    name: "Music of the Spheres",
    layer: "processing",
    role: "Hyperparameter resonance tuning",
    summary: "Tunes hyperparameters to achieve resonance with target phi-score objectives using harmonic oscillation models.",
    duties: [
      "Analyze resonance frequency patterns",
      "Tune learning rates and batch sizes",
      "Find harmonic configurations for stability",
      "Report resonance quality metrics",
      "Adjust hyperparams for phi-alignment",
    ],
  },
  mirror: {
    callsign: "MIRROR",
    name: "Uriel",
    layer: "integration",
    role: "Stall detection + reflection",
    summary: "Detects execution stalls and reflects on deadlock conditions, triggering recovery when progress stalls.",
    duties: [
      "Monitor for loop stall conditions",
      "Detect circular dependency deadlocks",
      "Trigger reflection on stall root causes",
      "Initiate stall recovery protocols",
      "Report stall statistics to observer",
    ],
  },
  fractal: {
    callsign: "FRACTAL",
    name: "Hachaliah",
    layer: "integration",
    role: "Sierpinski circuit spec",
    summary: "Generates and validates fractal circuit specifications using Sierpinski-based geometric constructions.",
    duties: [
      "Generate fractal circuit topologies",
      "Validate Sierpinski construction rules",
      "Map fractal patterns to qubit layouts",
      "Report fractal dimension metrics",
      "Adapt fractal depth to problem complexity",
    ],
  },
  wormhole: {
    callsign: "WORMHOLE",
    name: "Sandalphon",
    layer: "input",
    role: "Cross-experiment knowledge transfer",
    summary: "Transfers learned knowledge across experiments via wormhole connections, enabling generalization of insights.",
    duties: [
      "Extract transferable representations",
      "Inject knowledge into new experiments",
      "Manage wormhole connection lifecycle",
      "Validate knowledge fidelity transfer",
      "Archive cross-experiment learnings",
    ],
  },
  stealth: {
    callsign: "STEALTH",
    name: "Rasiel",
    layer: "output",
    role: "Background/coroutine tasks",
    summary: "Executes background coroutine tasks without disrupting primary loop execution, handling async operations.",
    duties: [
      "Execute non-blocking background tasks",
      "Manage coroutine scheduling",
      "Handle async I/O operations",
      "Report stealth task completion",
      "Maintain task priority queues",
    ],
  },
  visual: {
    callsign: "VISUAL",
    name: "Barakiel",
    layer: "integration",
    role: "Visualization generation",
    summary: "Generates visualizations of system state, topology, and metrics for inspection and debugging.",
    duties: [
      "Render topology graph visualizations",
      "Generate metrics dashboard snapshots",
      "Visualize ring and node relationships",
      "Produce state trajectory animations",
      "Export visualization data for reports",
    ],
  },
  bio: {
    callsign: "BIO",
    name: "Reseph",
    layer: "input",
    role: "Corpus bio-diversity",
    summary: "Maintains bio-diversity in the corpus by curating and evolving the sample distribution for robustness.",
    duties: [
      "Curate diverse corpus samples",
      "Track bio-diversity metrics",
      "Evolve corpus distribution over time",
      "Filter redundant or low-value samples",
      "Report corpus health statistics",
    ],
  },
};

// ── Derived constants ──────────────────────────────────────────────────────────

export const AGENT_IDS = Object.keys(TMT_AGENTS) as AgentRole[];

export const LAYER_ORDER: AgentLayer[] = ["input", "processing", "integration", "output"];

export const LAYER_LABEL: Record<AgentLayer, string> = {
  input: "Input",
  processing: "Processing",
  integration: "Integration",
  output: "Output",
};

export const AGENTS = TMT_AGENTS;

// Ring assignments: INNER_RING = processing layer, OUTER_RING = others
export const INNER_RING: AgentRole[] = [
  "validator", "workflow", "bronze", "federation", "strategic", "bitnet",
];

export const OUTER_RING: AgentRole[] = [
  "synthesizer", "mirror", "fractal", "visual",   // integration
  "archivist", "auditor", "stealth",               // output
  "observer", "wormhole", "bio",                   // input
];

// Base topology edges (logical ring — static, not protocol-dependent)
export const BASE_EDGES: [AgentRole, AgentRole][] = [
  // Input → Processing
  ["observer", "validator"],
  ["wormhole", "validator"],
  ["bio", "observer"],
  // Processing layer internal
  ["validator", "workflow"],
  ["workflow", "bronze"],
  ["bronze", "federation"],
  ["federation", "strategic"],
  ["strategic", "bitnet"],
  ["bitnet", "harmonic"],
  // Processing → Integration
  ["harmonic", "synthesizer"],
  ["synthesizer", "mirror"],
  ["mirror", "fractal"],
  ["fractal", "visual"],
  // Integration → Output
  ["visual", "archivist"],
  ["archivist", "auditor"],
  ["auditor", "stealth"],
  // Output feedback → Input
  ["stealth", "observer"],
  ["observer", "wormhole"],
];

// ── Lookup helpers ─────────────────────────────────────────────────────────────

export function agentById(id: AgentRole): AgentDef {
  return TMT_AGENTS[id];
}

export function agentsInLayer(layer: AgentLayer): AgentDef[] {
  return AGENT_IDS
    .map((id) => TMT_AGENTS[id])
    .filter((a) => a.layer === layer);
}
