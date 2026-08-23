import { useQuorum } from "@/lib/quorum/store";
import { engine } from "@/lib/quorum/engine";
import { AGENTS } from "@/lib/quorum/agents";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import type { AgentRole } from "@/lib/quorum/types";

const STATUS_VARIANT = {
  idle: "muted",
  sync: "ok",
  active: "default",
  wait: "warn",
  fault: "fault",
} as const;

function fmtHz(hz: number) {
  if (hz >= 1000) return `${(hz / 1000).toFixed(1)} kHz`;
  return `${hz.toFixed(0)} Hz`;
}

export function Inspector() {
  const selectedId = useQuorum((s) => s.selectedId);
  const agents = useQuorum((s) => s.agents);
  const invoking = useQuorum((s) => s.invoking);
  const actions = useQuorum((s) => s.actions);
  const benchmark = useQuorum((s) => s.benchmark);
  const version = useQuorum((s) => s.version);
  void version;

  const protocol = engine.protocol;
  const agent = selectedId ? AGENTS[selectedId] : null;
  const runtime = selectedId ? engine.runtime[selectedId] : null;

  // If agent is selected, show its live profile from store
  const liveProfile = selectedId
    ? agents.find((a) => a.agent_role === selectedId)
    : null;

  if (selectedId && liveProfile) {
    return (
      <div className="flex h-full flex-col gap-4">
        <div>
          <div className="flex items-center justify-between gap-3">
            <p className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">
              {liveProfile.layer} · {liveProfile.specialization}
            </p>
            <Badge variant={runtime?.status === "active" ? "default" : "muted"}>
              {runtime?.status ?? "idle"}
            </Badge>
          </div>
          <h2 className="mt-2 font-display text-3xl leading-none tracking-tight">
            {liveProfile.agent_name}
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {liveProfile.agent_role}
          </p>
        </div>
        {agent && <p className="text-sm leading-relaxed text-foreground/85">{agent.summary}</p>}
        <div className="grid grid-cols-3 gap-2">
          <Stat label="φ-Score" value={liveProfile.phi_score.toFixed(3)} />
          <Stat label="Resonance" value={fmtHz(liveProfile.resonance_frequency)} />
          <Stat label="Fitness" value={`${(liveProfile.fitness * 100).toFixed(1)}%`} />
          <Stat label="Load" value={liveProfile.current_load.toFixed(2)} />
          <Stat label="Success" value={`${(liveProfile.success_rate * 100).toFixed(0)}%`} />
          <Stat label="Φ-Align" value={liveProfile.phi_alignment.toFixed(3)} />
        </div>
        <Separator />
        <div>
          <p className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">Duties</p>
          <ul className="mt-2 space-y-1.5 text-sm text-foreground/80">
            {agent?.duties.map((d) => (
              <li key={d} className="pl-3 -indent-3 before:mr-2 before:text-subtle before:content-['—']">
                {d}
              </li>
            ))}
          </ul>
        </div>
        <div className="mt-auto space-y-2">
          <button
            className="w-full rounded-lg bg-primary px-4 py-2 text-sm font-medium disabled:opacity-50"
            disabled={invoking}
            onClick={() => actions.invoke(liveProfile.agent_id, `Execute ${selectedId} task`)}
          >
            {invoking ? "Invoking…" : "Invoke Agent"}
          </button>
          {benchmark && (
            <p className="text-center font-mono text-[11px] text-muted-foreground">
              benchmark: {benchmark.total_tasks} tasks · {(benchmark.success_rate * 100).toFixed(0)}% success
            </p>
          )}
        </div>
      </div>
    );
  }

  if (protocol) {
    return (
      <div className="flex h-full flex-col gap-4">
        <div>
          <p className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">
            Live protocol
          </p>
          <h2 className="mt-2 font-display text-3xl leading-none tracking-tight">{protocol}</h2>
        </div>
        <Separator />
        <LogList />
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-4">
      <div>
        <p className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">
          Lattice
        </p>
        <h2 className="mt-2 font-display text-3xl leading-none tracking-tight">
          17 agents
        </h2>
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
          Select a node on the lattice to inspect its profile, φ-score, and duties.
          Use the invoke button to dispatch an agent task.
        </p>
      </div>
      <Separator />
      <LogList />
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-secondary px-3 py-2">
      <p className="font-mono text-[10px] tracking-[0.14em] text-muted-foreground uppercase">{label}</p>
      <p className="mt-1 font-mono text-sm tabular-nums">{value}</p>
    </div>
  );
}

function LogList() {
  const version = useQuorum((s) => s.version);
  void version;
  const events = engine.log.slice(-8).reverse();
  return (
    <div>
      <p className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">Log</p>
      <ul className="mt-2 space-y-2">
        {events.length === 0 && (
          <li className="text-sm text-muted-foreground">No events yet.</li>
        )}
        {events.map((e, i) => (
          <li key={`${e.t}-${i}`} className="text-sm">
            <span className="font-mono text-[11px] text-subtle">
              {e.agent ? AGENTS[e.agent as AgentRole]?.callsign ?? e.agent : "SYS"}
            </span>
            <span className="mt-0.5 block text-foreground/80">{e.text}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
