import { create } from "zustand";
import type {
  AgentProfile,
  AgentRole,
  TelemetryData,
  TopologyData,
  ProtocolsResponse,
  BenchmarkResults,
  InvokeResult,
} from "./types";

// ── API client ─────────────────────────────────────────────────────────────────

const BASE = "/copilot";

export async function apiFetchAgents(): Promise<AgentProfile[]> {
  const r = await fetch(`${BASE}/agents`);
  if (!r.ok) throw new Error(await r.text());
  const data = await r.json() as { agents: AgentProfile[]; total: number };
  return data.agents;
}

export async function apiFetchAgent(id: number): Promise<AgentProfile> {
  const r = await fetch(`${BASE}/agents/${id}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function apiInvokeAgent(
  id: number,
  objective: string,
  taskType = "validation",
  executionMode = "simulation"
): Promise<InvokeResult> {
  const r = await fetch(`${BASE}/agents/${id}/invoke`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ objective, task_type: taskType, context: {}, execution_mode: executionMode }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function apiFetchTopology(): Promise<TopologyData> {
  const r = await fetch(`${BASE}/topology`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function apiFetchProtocols(): Promise<ProtocolsResponse> {
  const r = await fetch(`${BASE}/protocols`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function apiFetchBenchmark(): Promise<BenchmarkResults> {
  const r = await fetch(`${BASE}/benchmark/results`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export function apiConnectTelemetry(
  onUpdate: (data: TelemetryData) => void
): () => void {
  const es = new EventSource(`${BASE}/telemetry`);
  es.onmessage = (e) => {
    try { onUpdate(JSON.parse(e.data) as TelemetryData); } catch { /* ignore */ }
  };
  es.onerror = () => es.close();
  return () => es.close();
}

// ── Store ─────────────────────────────────────────────────────────────────────

interface State {
  agents: AgentProfile[];
  selectedId: AgentRole | null;
  briefing: boolean;
  telemetry: TelemetryData | null;
  benchmark: BenchmarkResults | null;
  topology: TopologyData | null;
  protocols: ProtocolsResponse | null;
  invoking: boolean;
  wsStatus: "disconnected" | "connecting" | "connected";
  error: string | null;
  version: number;
}

interface Actions {
  fetchAgents: () => Promise<void>;
  fetchTopology: () => Promise<void>;
  fetchProtocols: () => Promise<void>;
  fetchBenchmark: () => Promise<void>;
  select: (id: AgentRole | null) => void;
  invoke: (id: number, objective: string) => Promise<InvokeResult | void>;
  setTelemetry: (data: TelemetryData) => void;
  setWsStatus: (s: "disconnected" | "connecting" | "connected") => void;
  bump: () => void;
  toggleRunning: () => void;
  dismissBriefing: () => void;
  showBriefing: () => void;
}

export const useQuorum = create<State & { actions: Actions }>((set, get) => ({
  agents: [],
  selectedId: null,
  briefing: false,
  telemetry: null,
  benchmark: null,
  topology: null,
  protocols: null,
  invoking: false,
  wsStatus: "disconnected",
  error: null,
  version: 0,

  actions: {
    async fetchAgents() {
      try {
        set({ error: null });
        const agents = await apiFetchAgents();
        set({ agents });
      } catch (err) {
        set({ error: String(err) });
      }
    },

    async fetchTopology() {
      try { set({ topology: await apiFetchTopology() }); }
      catch (err) { set({ error: String(err) }); }
    },

    async fetchProtocols() {
      try { set({ protocols: await apiFetchProtocols() }); }
      catch (err) { set({ error: String(err) }); }
    },

    async fetchBenchmark() {
      try { set({ benchmark: await apiFetchBenchmark() }); }
      catch (err) { set({ error: String(err) }); }
    },

    select(id) {
      const { engine } = require("@/lib/quorum/engine");
      engine.selectedId = id;
      set({ selectedId: id, version: get().version + 1 });
    },

    async invoke(id, objective) {
      set({ invoking: true, error: null });
      try {
        const result = await apiInvokeAgent(id, objective);
        set({ version: get().version + 1 });
        return result;
      } catch (err) {
        set({ error: String(err) });
      } finally {
        set({ invoking: false });
      }
    },

    setTelemetry(data) {
      set({ telemetry: data, version: get().version + 1 });
    },

    setWsStatus(s) { set({ wsStatus: s }); },

    bump() { set({ version: get().version + 1 }); },

    toggleRunning() {
      const { engine } = require("@/lib/quorum/engine");
      engine.running = !engine.running;
      set({ version: get().version + 1 });
    },

    dismissBriefing() {
      try { window.localStorage.setItem("tinymetatron.briefing", "1"); } catch { /* ignore */ }
      set({ briefing: false });
    },

    showBriefing() { set({ briefing: true }); },
  },
}));

// Init: fetch all static data + SSE
export function initStore() {
  const { actions } = useQuorum.getState() as State & { actions: Actions };
  actions.fetchAgents();
  actions.fetchTopology();
  actions.fetchProtocols();
  actions.fetchBenchmark();
  apiConnectTelemetry((data) => actions.setTelemetry(data));
}
