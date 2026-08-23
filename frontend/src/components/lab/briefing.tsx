import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { useQuorum } from "@/lib/quorum/store";

export function Briefing() {
  const open = useQuorum((s) => s.briefing);
  const dismiss = useQuorum((s) => s.dismissBriefing);
  useEffect(() => {
    try {
      if (window.localStorage.getItem("tinymetatron.briefing") !== "1") {
        useQuorum.getState().showBriefing();
      }
    } catch {
      useQuorum.getState().showBriefing();
    }
  }, []);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-40 flex items-end justify-center bg-background/80 p-4 sm:items-center">
      <div className="w-full max-w-lg rounded-2xl bg-card p-6 shadow-[var(--shadow-border)] sm:p-8">
        <p className="font-mono text-[10px] tracking-[0.2em] text-muted-foreground uppercase">
          System briefing
        </p>
        <h1 className="mt-3 font-display text-4xl leading-none tracking-tight sm:text-5xl">
          TinyMetatron Copilot
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          17 agents. 4 layers. Real loop execution.
        </p>
        <p className="mt-5 text-sm leading-relaxed text-foreground/85">
          A multi-agent orchestration lab backed by TinyMetatron loops. Agents execute
          training, corpus, and evaluation pipelines through the{" "}
          <code className="font-mono text-xs">/copilot/*</code> API endpoints.
        </p>
        <ul className="mt-5 space-y-2 text-sm text-muted-foreground">
          <li>Execution modes: simulation (dry-run), live (real loops), hybrid (quantum fallback)</li>
          <li>φ-score governs agent selection and routing</li>
          <li>Sierpinski fractal topology: 13 nodes across 5 rings</li>
        </ul>
        <div className="mt-8 flex justify-end">
          <Button onClick={dismiss}>Enter the lab</Button>
        </div>
      </div>
    </div>
  );
}
