import { createFileRoute } from "@tanstack/react-router";
import { Shell } from "@/components/lab/shell";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useEffect, useState } from "react";
import { apiFetchProtocols, apiFetchBenchmark } from "@/lib/quorum/store";
import type { ProtocolsResponse, BenchmarkResults } from "@/lib/quorum/types";

export const Route = createFileRoute("/protocols")({ component: ProtocolsPage });

function ProtocolsPage() {
  const [protocols, setProtocols] = useState<ProtocolsResponse | null>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkResults | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([apiFetchProtocols(), apiFetchBenchmark()])
      .then(([p, b]) => { setProtocols(p); setBenchmark(b); })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return (
    <Shell>
      <ScrollArea className="h-full">
        <div className="mx-auto w-full max-w-3xl px-4 py-6 lg:px-8">
          <p className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">
            Execution
          </p>
          <h1 className="mt-2 font-display text-4xl tracking-tight">Protocols</h1>
          <p className="mt-2 max-w-xl text-sm leading-relaxed text-muted-foreground">
            Three execution modes control how the 17-agent copilot interacts with TinyMetatron loops.
            Simulation runs agents in dry-run without real training. Live executes real loops.
            Hybrid adds IBM Quantum hardware fallback.
          </p>

          {loading ? (
            <p className="mt-8 text-sm text-muted-foreground">Loading…</p>
          ) : (
            <>
              <section className="mt-8">
                <h2 className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">
                  Execution Modes
                </h2>
                <div className="mt-4 space-y-3">
                  {protocols?.protocols.map((p) => (
                    <div
                      key={p.value}
                      className="rounded-xl bg-card p-5 shadow-[var(--shadow-border)]"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <h3 className="font-display text-xl leading-none">{p.name}</h3>
                          <p className="mt-1 text-sm text-muted-foreground">{p.description}</p>
                        </div>
                        <span className="rounded-full bg-secondary px-2.5 py-0.5 font-mono text-xs">
                          {p.value}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              {benchmark && (
                <section className="mt-10">
                  <h2 className="font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">
                    Latest Benchmark
                  </h2>
                  <div className="mt-4 grid grid-cols-3 gap-3">
                    <Stat label="Total Tasks" value={String(benchmark.total_tasks)} />
                    <Stat label="Success Rate" value={`${(benchmark.success_rate * 100).toFixed(1)}%`} />
                    <Stat label="Agreement" value={`${(benchmark.agreement_rate * 100).toFixed(1)}%`} />
                    <Stat label="Coordination Score" value={benchmark.coordination_quality_score.toFixed(3)} />
                    <Stat label="Duration" value={`${benchmark.duration_seconds.toFixed(2)}s`} />
                    <Stat label="Failed" value={String(benchmark.failed_tasks)} />
                  </div>

                  <h3 className="mt-6 font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">
                    Per Task Type
                  </h3>
                  <div className="mt-3 space-y-2">
                    {Object.entries(benchmark.task_results).map(([type, result]) => (
                      <div key={type} className="flex items-center justify-between rounded-lg bg-secondary px-4 py-2">
                        <span className="font-mono text-sm">{type}</span>
                        <span className="font-mono text-xs text-muted-foreground">
                          {result.successful}/{result.iterations} · {result.avg_latency_ms.toFixed(1)}ms avg
                        </span>
                      </div>
                    ))}
                  </div>
                </section>
              )}
            </>
          )}
        </div>
      </ScrollArea>
    </Shell>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-card p-4 shadow-[var(--shadow-border)]">
      <p className="font-mono text-[10px] tracking-[0.14em] text-muted-foreground uppercase">{label}</p>
      <p className="mt-1 font-mono text-lg tabular-nums">{value}</p>
    </div>
  );
}
