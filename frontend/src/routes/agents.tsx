import { createFileRoute } from "@tanstack/react-router";
import { AgentList } from "@/components/lab/agent-list";
import { Inspector } from "@/components/lab/inspector";
import { Shell } from "@/components/lab/shell";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AGENTS, AGENT_IDS, LAYER_LABEL, LAYER_ORDER } from "@/lib/quorum/agents";
import { engine } from "@/lib/quorum/engine";
import { useQuorum } from "@/lib/quorum/store";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/agents")({ component: AgentsPage });

function AgentsPage() {
  const select = useQuorum((s) => s.select);
  const selectedId = useQuorum((s) => s.selectedId);
  const version = useQuorum((s) => s.version);
  void version;

  return (
    <Shell>
      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[minmax(0,1fr)_320px]">
        <ScrollArea className="h-full">
          <div className="mx-auto w-full max-w-5xl px-4 py-6 lg:px-8">
            <p className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">
              Roster
            </p>
            <h1 className="mt-2 font-display text-4xl tracking-tight">Seventeen agents</h1>
            <p className="mt-2 max-w-xl text-sm leading-relaxed text-muted-foreground">
              Four layers. No shared heap. Each agent is specified by identity, duties, and the
              neighbors it is allowed to address.
            </p>
            <div className="mt-8 space-y-8">
              {LAYER_ORDER.map((layer) => (
                <section key={layer}>
                  <h2 className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">
                    {LAYER_LABEL[layer]}
                  </h2>
                  <div className="mt-3 grid gap-3 sm:grid-cols-2">
                    {AGENT_IDS.filter((id) => AGENTS[id].layer === layer).map((id) => {
                      const a = AGENTS[id];
                      const r = engine.runtime[id];
                      const active = selectedId === id;
                      return (
                        <button
                          key={id}
                          type="button"
                          onClick={() => select(id)}
                          className={cn(
                            "rounded-xl p-4 text-left shadow-[var(--shadow-border)] transition-[box-shadow,background-color] duration-150",
                            active ? "bg-secondary" : "bg-card hover:shadow-[var(--shadow-border-hover)]",
                          )}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <p className="font-mono text-[10px] text-subtle">{a.code}</p>
                              <p className="mt-1 font-display text-2xl leading-none">{a.callsign}</p>
                              <p className="mt-1 text-sm text-muted-foreground">{a.name}</p>
                            </div>
                            <Badge variant={r.status === "active" ? "default" : "muted"}>{r.status}</Badge>
                          </div>
                          <p className="mt-3 line-clamp-2 text-sm leading-relaxed text-foreground/80">
                            {a.summary}
                          </p>
                          <p className="mt-3 font-mono text-[11px] tabular-nums text-muted-foreground">
                            {(r.fidelity * 100).toFixed(2)}% · {a.qubits} q
                          </p>
                        </button>
                      );
                    })}
                  </div>
                </section>
              ))}
            </div>
          </div>
        </ScrollArea>
        <aside className="hidden border-l border-border lg:block">
          <ScrollArea className="h-full">
            <div className="p-5">
              {selectedId ? <Inspector /> : <AgentList compact />}
            </div>
          </ScrollArea>
        </aside>
      </div>
    </Shell>
  );
}
