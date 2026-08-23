import type { AgentRole, Metrics, LogEvent } from "./types";
import { BASE_EDGES } from "./agents";

export interface Engine {
  running: boolean;
  time: number;
  protocol: string;
  selectedId: AgentRole | null;
  hoveredId: AgentRole | null;
  runtime: Record<AgentRole, number>; // agent_role → load 0–1
  metrics: Metrics;
  log: LogEvent[];
  wsStatus: "disconnected" | "connecting" | "connected";
}

export const INITIAL_METRICS: Metrics = {
  fidelity: 0,
  bonds: 0,
  coherenceNs: 0,
  eventRate: 0,
};

export const engine: Engine = {
  running: true,
  time: 0,
  protocol: "",
  selectedId: null,
  hoveredId: null,
  runtime: {} as Record<AgentRole, number>,
  metrics: { ...INITIAL_METRICS },
  log: [],
  wsStatus: "disconnected",
};

let ws: WebSocket | null = null;
let wsRetries = 0;
let wsTimer: ReturnType<typeof setTimeout> | null = null;
let pingTimer: ReturnType<typeof setInterval> | null = null;

const MAX_RETRIES = 5;
const BASE_DELAY = 1000;

function wsUrl(): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/copilot/ws`;
}

function clearPing(): void {
  if (pingTimer) {
    clearInterval(pingTimer);
    pingTimer = null;
  }
}

function scheduleReconnect(): void {
  if (wsRetries >= MAX_RETRIES) return;
  const delay = BASE_DELAY * Math.pow(2, wsRetries);
  wsRetries++;
  wsTimer = setTimeout(wsConnect, delay);
}

export function wsConnect(): void {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

  engine.wsStatus = "connecting";
  ws = new WebSocket(wsUrl());

  ws.onopen = () => {
    wsRetries = 0;
    engine.wsStatus = "connected";
    ws!.send(JSON.stringify({ type: "ping" }));
    pingTimer = setInterval(() => {
      if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping" }));
      }
    }, 15000);
  };

  ws.onmessage = (e: MessageEvent) => {
    try {
      const msg = JSON.parse(e.data) as {
        type: string;
        agent_role?: string;
        status?: string;
        confidence?: number;
      };

      if (msg.type === "result" && msg.agent_role) {
        const role = msg.agent_role as AgentRole;
        const confidence = msg.confidence ?? 0;
        engine.runtime[role] = 1 - confidence;
        engine.protocol = role;
        engine.log.unshift({
          t: Date.now(),
          agent: role,
          text: `[${role}] status=${msg.status ?? "unknown"} confidence=${(confidence * 100).toFixed(1)}%`,
        });
        if (engine.log.length > 200) engine.log.pop();
      }
    } catch {
      /* ignore */
    }
  };

  ws.onclose = () => {
    engine.wsStatus = "disconnected";
    clearPing();
    scheduleReconnect();
  };

  ws.onerror = () => ws?.close();
}

export function disconnectWs(): void {
  if (wsTimer) clearTimeout(wsTimer);
  clearPing();
  ws?.close();
  ws = null;
  engine.wsStatus = "disconnected";
}

export function applyTelemetry(data: {
  agent_utilization: Record<string, number>;
  phi_alignment_rate: number;
  agreement_rate: number;
  tasks_completed: number;
}): void {
  for (const [role, load] of Object.entries(data.agent_utilization)) {
    engine.runtime[role as AgentRole] = load;
  }
  engine.metrics = {
    fidelity: data.phi_alignment_rate,
    bonds: data.agreement_rate,
    coherenceNs: 0,
    eventRate: data.tasks_completed,
  };
}

export function tick(dt: number): void {
  if (engine.running) {
    engine.time += dt;
  }
}

export function selectAgent(id: AgentRole | null): void {
  engine.selectedId = id;
}

export function hoverAgent(id: AgentRole | null): void {
  engine.hoveredId = id;
}

export function runProtocol(role: AgentRole): void {
  if (ws?.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({
    type: "invoke",
    agent_role: role,
    objective: `Execute ${role} task`,
    task_type: "validation",
    context: {},
    execution_mode: "simulation",
  }));
  engine.protocol = role;
}

export function haltProtocol(): void {
  engine.protocol = "";
}

export function liveEdges(): [AgentRole, AgentRole][] {
  return BASE_EDGES;
}
