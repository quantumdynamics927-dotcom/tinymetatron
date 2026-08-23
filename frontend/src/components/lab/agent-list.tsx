import { AGENTS, LAYER_ORDER } from "@/lib/quorum/agents";
import { engine } from "@/lib/quorum/engine";
import { useQuorum } from "@/lib/quorum/store";
import type { AgentRole } from "@/lib/quorum/types";
import { cn } from "@/lib/utils";

export function AgentList({ compact = false }: { compact?: boolean }) {
  const selectedId = useQuorum((s) => s.selectedId);
  const actions = useQuorum((s) => s.actions);
  const agents = useQuorum((s) => s.agents);
  const version = useQuorum((s) => s.version);
  void version;

  // Fallback to static AGENTS if agents not loaded yet
  const displayAgents = agents.length > 0 ? agents : null;

  return (
    <div className="flex flex-col gap-5">
      {LAYER_ORDER.map((layer) => (
        <section key={layer}>
          <h3 className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">
            {layer}
          </h3>
          <ul className="mt-2 flex flex-col gap-0.5">
            {(displayAgents ?? Object.values(AGENTS)).map((a) => {
              const id = (a.agent_role ?? a.id) as AgentRole;
              const def = AGENTS[id] ?? { callsign: String(id), role: "", code: "" };
              const r = engine.runtime[id] ?? { status: "idle", load: 0 };
              const active = selectedId === id;
              return (
                <li key={id}>
                  <button
                    type="button"
                    onClick={() => actions.select(active ? null : id)}
                    className={cn(
                      "flex w-full items-center gap-3 rounded-md px-2 py-2 text-left transition-colors duration-150",
                      active ? "bg-secondary text-foreground" : "text-foreground/85 hover:bg-accent",
                    )}
                  >
                    <StatusDot status={r.status} />
                    <span className="min-w-0 flex-1">
                      <span className="block font-mono text-xs tracking-wide">{def.callsign}</span>
                      {!compact && (
                        <span className="block truncate text-xs text-muted-foreground">{def.role}</span>
                      )}
                    </span>
                    <span className="font-mono text-[10px] text-subtle">{def.code || id.slice(0, 4).toUpperCase()}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      ))}
    </div>
  );
}

function StatusDot({ status }: { status: string }) {
  const color =
    status === "active" ? "bg-primary"
    : status === "sync" ? "bg-ok"
    : status === "wait" ? "bg-warn"
    : status === "fault" ? "bg-fault"
    : "bg-subtle";
  return <span className={cn("size-1.5 shrink-0 rounded-full", color)} aria-hidden="true" />;
}
