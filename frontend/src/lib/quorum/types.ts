// ── Agent Role & Layer (matches Python AgentRole / AgentLayer) ──────────────────
export type AgentRole =
  | "validator" | "synthesizer" | "workflow" | "observer"
  | "archivist" | "auditor" | "bronze" | "federation"
  | "strategic" | "bitnet" | "harmonic" | "mirror"
  | "fractal" | "wormhole" | "stealth" | "visual" | "bio";

export type AgentLayer = "input" | "processing" | "integration" | "output";

export type ExecutionMode = "simulation" | "live" | "hybrid";

// ── API response shapes ────────────────────────────────────────────────────────
export interface AgentProfile {
  agent_id: number;
  agent_name: string;
  agent_role: AgentRole;
  layer: AgentLayer;
  specialization: string;
  phi_score: number;          // 0–1 golden ratio alignment
  resonance_frequency: number; // Hz
  fitness: number;            // 0–1
  current_load: number;
  availability: number;
  phi_alignment: number;
  success_rate: number;
}

export interface AgentListResponse {
  agents: AgentProfile[];
  total: number;
}

export interface InvokeResult {
  agent_id: number;
  agent_role: string;
  status: string;
  confidence: number;
  trace_id: string;
  session_id: string;
  total_duration_ms: number;
  contracts: number;
  decisions: number;
}

export interface TopologyData {
  nodes: number;
  rings: Record<string, {
    nodes: string[];
    radius: number;
    phase_offset: number;
  }>;
  scaling_factor: number;
  qubit_map: Record<number, number>;
}

export interface ProtocolInfo {
  name: string;
  value: string;
  description: string;
}

export interface ProtocolsResponse {
  protocols: ProtocolInfo[];
  modes: ExecutionMode[];
}

export interface TelemetryData {
  timestamp: number;
  session_id: string;
  tasks_completed: number;
  tasks_failed: number;
  agreement_rate: number;
  phi_alignment_rate: number;
  agent_utilization: Record<string, number>;
}

export interface BenchmarkTaskResult {
  task_type: string;
  iterations: number;
  successful: number;
  failed: number;
  avg_latency_ms: number;
  avg_confidence: number;
  success_rate: number;
}

export interface BenchmarkResults {
  benchmark_id: string;
  duration_seconds: number;
  total_tasks: number;
  successful_tasks: number;
  failed_tasks: number;
  success_rate: number;
  agreement_rate: number;
  coordination_quality_score: number;
  task_results: Record<string, BenchmarkTaskResult>;
}

// ── UI state types (mirror Quorum shapes for component compatibility) ──────────
export type AgentStatus = "idle" | "active" | "sync" | "wait" | "fault";

export interface AgentRuntime {
  status: AgentStatus;
  fidelity: number;    // mapped from phi_score
  load: number;        // current_load
  coherenceNs: number; // derived from resonance_frequency
  lastEvent: string;
  messagesIn: number;
  messagesOut: number;
}

export interface LogEvent {
  t: number;
  agent?: AgentRole;
  text: string;
}

export interface Metrics {
  fidelity: number;
  bonds: number;
  coherenceNs: number;
  eventRate: number;
}

export interface History {
  fidelity: number[];
  bonds: number[];
  coherence: number[];
  rate: number[];
}

export const HISTORY_LEN = 96;
