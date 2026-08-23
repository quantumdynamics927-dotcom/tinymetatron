import { useEffect } from "react";
import { initStore } from "@/lib/quorum/store";
import { useQuorum } from "@/lib/quorum/store";

function Spark({ data, label, value }: { data: number[]; label: string; value: string }) {
  const w = 120;
  const h = 28;
  const min = Math.min(...data, 0);
  const max = Math.max(...data, 1);
  const span = max - min || 1;
  const d = data
    .map((n, i) => {
      const x = data.length <= 1 ? 0 : (i / (data.length - 1)) * w;
      const y = h - ((n - min) / span) * (h - 4) - 2;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <div className="flex min-w-0 items-center gap-3">
      <div className="min-w-0">
        <p className="font-mono text-[10px] tracking-[0.14em] text-muted-foreground uppercase">{label}</p>
        <p className="font-mono text-sm tabular-nums text-foreground">{value}</p>
      </div>
      <svg width={w} height={h} className="hidden shrink-0 sm:block" aria-hidden="true">
        {d ? (
          <path d={d} fill="none" stroke="currentColor" strokeWidth="1.2" className="text-primary/80" />
        ) : null}
      </svg>
    </div>
  );
}

export function TelemetryStrip() {
  const telemetry = useQuorum((s) => s.telemetry);
  const wsStatus = useQuorum((s) => s.wsStatus);

  return (
    <div className="grid grid-cols-2 gap-x-6 gap-y-3 border-t border-border px-4 py-3 lg:grid-cols-4 lg:px-6">
      <Spark
        data={telemetry ? [telemetry.phi_alignment_rate] : []}
        label="φ-Alignment"
        value={telemetry ? `${(telemetry.phi_alignment_rate * 100).toFixed(1)}%` : "—"}
      />
      <Spark
        data={telemetry ? [telemetry.agreement_rate] : []}
        label="Agreement"
        value={telemetry ? `${(telemetry.agreement_rate * 100).toFixed(1)}%` : "—"}
      />
      <Spark
        data={telemetry ? [telemetry.tasks_completed] : []}
        label="Tasks"
        value={telemetry ? String(telemetry.tasks_completed) : "0"}
      />
      <div className="flex min-w-0 items-center gap-3">
        <div className="min-w-0">
          <p className="font-mono text-[10px] tracking-[0.14em] text-muted-foreground uppercase">WS</p>
          <p className={`font-mono text-sm tabular-nums ${
            wsStatus === "connected" ? "text-ok" : wsStatus === "connecting" ? "text-warn" : "text-muted-foreground"
          }`}>
            {wsStatus}
          </p>
        </div>
      </div>
    </div>
  );
}
