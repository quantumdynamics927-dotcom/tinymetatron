import { useState, useEffect } from "react";
import type { ReactNode } from "react";
import { Link } from "@tanstack/react-router";
import { Pause, Play, Square } from "lucide-react";
import { Briefing } from "@/components/lab/briefing";
import { TelemetryStrip } from "@/components/lab/telemetry";
import { Button } from "@/components/ui/button";
import { engine, haltProtocol, runProtocol } from "@/lib/quorum/engine";
import { useQuorum } from "@/lib/quorum/store";
import type { AgentRole } from "@/lib/quorum/types";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Lattice" },
  { to: "/agents", label: "Agents" },
  { to: "/protocols", label: "Protocols" },
  { to: "/repository", label: "Repository" },
] as const;

const AGENT_ROLES: AgentRole[] = [
  "validator", "synthesizer", "workflow", "observer", "archivist",
  "auditor", "bronze", "federation", "strategic", "bitnet",
  "harmonic", "mirror", "fractal", "wormhole", "stealth", "visual", "bio",
];

export function Shell({ children }: { children: ReactNode }) {
  const [pathname, setPathname] = useState("/");
  useEffect(() => { setPathname(window.location.pathname); }, []);
  const bump = useQuorum((s) => s.actions.bump);
  const live = Boolean(engine.protocol);

  const toggle = () => { engine.running = !engine.running; bump(); };
  const halt = () => { haltProtocol(); bump(); };
  const handleRun = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const role = e.target.value as AgentRole;
    if (role) { runProtocol(role); bump(); }
    e.target.selectedIndex = 0;
  };

  return (
    <div className="flex min-h-dvh flex-col bg-background text-foreground">
      <Briefing />
      <header className="flex flex-col gap-3 border-b border-border px-4 py-3 sm:flex-row sm:items-center sm:justify-between lg:px-6">
        <div className="flex items-center justify-between gap-4">
          <Link to="/" className="flex items-baseline gap-2">
            <span className="font-display text-2xl leading-none tracking-tight">TMT</span>
            <span className="hidden font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase sm:inline">
              Copilot v2
            </span>
          </Link>
          <span className={cn(
            "font-mono text-[10px] tracking-[0.16em] uppercase sm:hidden",
            live ? "text-ok" : "text-muted-foreground",
          )}>
            {live ? "Live" : "Idle"}
          </span>
        </div>
        <nav className="flex items-center gap-1 overflow-x-auto">
          {NAV.map((item) => {
            const active = pathname === item.to;
            return (
              <Link
                key={item.to}
                to={item.to}
                className={cn(
                  "rounded-md px-3 py-2 text-sm transition-colors duration-150",
                  active ? "bg-secondary text-foreground" : "text-muted-foreground hover:text-foreground",
                )}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="flex flex-wrap items-center gap-2">
          <select
            id="agent-select"
            className="h-10 min-h-10 rounded-md bg-secondary px-3 text-sm text-foreground shadow-[var(--shadow-border)]"
            defaultValue=""
            onChange={handleRun}
          >
            <option value="" disabled>Invoke agent</option>
            {AGENT_ROLES.map((role) => (
              <option key={role} value={role}>{role}</option>
            ))}
          </select>
          <Button variant="outline" size="icon" onClick={toggle} aria-label={engine.running ? "Pause" : "Resume"}>
            {engine.running ? <Pause /> : <Play />}
          </Button>
          <Button variant="ghost" size="icon" onClick={halt} disabled={!live} aria-label="Halt">
            <Square />
          </Button>
          <span className={cn(
            "hidden font-mono text-[10px] tracking-[0.16em] uppercase sm:inline",
            live ? "text-ok" : "text-muted-foreground",
          )}>
            {live ? "Live" : "Idle"}
          </span>
        </div>
      </header>
      <div className="flex min-h-0 flex-1 flex-col">{children}</div>
      <TelemetryStrip />
    </div>
  );
}
